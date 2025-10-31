# agents/ace.py
"""
Agentic Context Engineering (ACE) Agent

Learns through context evolution rather than parametric belief updates.
Maintains a structured "playbook" of strategies that grows through
reflection and curation after each episode.

Based on: "Agentic Context Engineering" paper
"""

import time
import json
import re
import uuid
from typing import Tuple, Optional, Dict, List, Any

from agents.base import Agent, AgentStep, LLMInterface
from experiments.prompts import (
    ACE_GENERATOR_TEMPLATE,
    ACE_REFLECTOR_TEMPLATE,
    ACE_CURATOR_TEMPLATE,
    ACE_QUERY_TEMPLATE,
    extract_answer_components,
    extract_action,
    extract_thought,
    extract_used_bullets,
    format_observation_history
)
from models.tools import get_tools_for_environment
from utils.token_accounting import TokenAccountant
from utils.embeddings import BulletRetriever


class ACEAgent(Agent):
    """
    Agent that learns through context evolution rather than belief updates.

    Maintains a structured "playbook" of strategies, updated through:
    - Reflector: Extracts insights from episode trajectories
    - Curator: Organizes insights into incremental context updates

    Architecture:
    1. Generator: Uses playbook to decide actions (like Actor)
    2. Reflector: Analyzes episode outcomes to extract insights
    3. Curator: Synthesizes insights into playbook updates
    """

    def __init__(
        self,
        llm: LLMInterface,
        action_budget: int,
        environment_name: Optional[str] = None,
        curation_mode: str = "curated",
        token_cap: Optional[int] = None,
        use_retrieval: bool = True,
        top_k: int = 5,
        reflection_rounds: int = 1,
        generator_temperature: float = 0.7,
        reflector_temperature: float = 0.7,
        curator_temperature: float = 0.7,
        max_epochs: int = 1
    ):
        """
        Initialize ACE agent.

        Args:
            llm: LLM interface
            action_budget: Maximum number of actions per episode
            environment_name: Name of environment (for tool selection)
            curation_mode: Curation strategy - "curated" (default), "no_curate", "random", "greedy"
            token_cap: Maximum playbook tokens (None = unlimited, 512/1k/2k for budget tests)
            use_retrieval: Whether to use top-k retrieval (True = faithful ACE)
            top_k: Number of bullets to retrieve per section
            reflection_rounds: Number of reflection iterations per episode (1+ for faithful ACE)
            generator_temperature: Temperature for Generator LLM calls (0.7 default for faithful ACE)
            reflector_temperature: Temperature for Reflector LLM calls (0.7 default for faithful ACE)
            curator_temperature: Temperature for Curator LLM calls (0.7 default for faithful ACE)
            max_epochs: Maximum number of offline epochs (replays same seeds to strengthen context)
        """
        super().__init__(llm, action_budget)
        self.environment_name = environment_name
        self.curation_mode = curation_mode
        self.token_cap = token_cap
        self.use_retrieval = use_retrieval
        self.top_k = top_k
        self.reflection_rounds = reflection_rounds
        self.max_epochs = max_epochs
        self.generator_temperature = generator_temperature
        self.reflector_temperature = reflector_temperature
        self.curator_temperature = curator_temperature
        self.playbook = self._init_playbook()
        self.episode_history = []
        self.current_epoch = 0  # Track which epoch we're on
        self.tools_class = None
        self.token_accountant = TokenAccountant()  # Track token breakdown

        # Initialize retriever for top-k bullet selection
        if use_retrieval:
            self.retriever = BulletRetriever(top_k=top_k)
        else:
            self.retriever = None

        if environment_name:
            try:
                self.tools_class = get_tools_for_environment(environment_name)
            except ValueError:
                print(f"Warning: No tools found for {environment_name}")

    def _init_playbook(self) -> Dict[str, List[Dict]]:
        """
        Initialize empty playbook with sections.

        Returns:
            Playbook dictionary with empty sections
        """
        return {
            'strategies_and_hard_rules': [],
            'useful_code_snippets': [],
            'troubleshooting_and_pitfalls': [],
            'apis_to_use': [],
            'verification_checklist': []
        }

    def reset(self):
        """
        Reset for new episode (playbook persists).

        Note: Unlike Actor agent, playbook is NOT reset between episodes.
        This allows the agent to accumulate knowledge over time.
        """
        super().reset()

        # Reset token accounting for new episode
        self.token_accountant.reset()

        # Start new episode history
        self.episode_history.append({
            'steps': [],
            'outcome': None,
            'playbook_snapshot': self._get_playbook_snapshot(),
            'referenced_bullets': []  # Track which bullets were used
        })

    def act(self, observation: dict) -> AgentStep:
        """
        Generate action using current playbook (Generator).

        Args:
            observation: Environment observation

        Returns:
            AgentStep with action and metadata
        """
        # Choose action if budget allows
        if self.action_count < self.action_budget:
            thought, action = self._choose_action(observation)
            self.action_count += 1
        else:
            thought = "Action budget exhausted"
            action = None

        # Create step record
        step = AgentStep(
            timestamp=time.time(),
            step_num=len(self.memory),
            thought=thought,
            action=action,
            observation=observation,
            belief_state={'playbook_size': self._get_playbook_size()},
            surprisal=0.0,
            token_usage=0
        )

        # Log step to episode history
        if self.episode_history:
            self.episode_history[-1]['steps'].append({
                'observation': observation,
                'action': action,
                'thought': thought,
                'playbook_size': self._get_playbook_size()
            })

        self.memory.append(step)
        return step

    def answer_query(self, question: str) -> Tuple[str, float]:
        """
        Answer test query using playbook.

        Args:
            question: Question to answer

        Returns:
            Tuple of (answer, confidence)
        """
        # Format playbook as context
        playbook_text = self._format_playbook()

        # Build query prompt
        prompt = ACE_QUERY_TEMPLATE.format(
            playbook=playbook_text,
            memory_summary=format_observation_history(self.memory, max_steps=10),
            question=question
        )

        # Query LLM
        response = self.llm.generate(prompt, temperature=0.0)

        # Record token usage for evaluation
        input_tokens, output_tokens = self.llm.get_last_usage()
        self.token_accountant.record(
            'evaluation',
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            metadata={'question': question[:50]}  # Truncate question
        )

        # Parse answer
        answer, confidence, reasoning = extract_answer_components(response)

        return answer, confidence

    def update_playbook(self, outcome: dict):
        """
        Update playbook after episode completion.

        This is the core learning mechanism for ACE agents.

        Args:
            outcome: Episode outcome including test results
        """
        if not self.episode_history:
            print("Warning: No episode history to update from")
            return

        # 1. Update bullet feedback counts based on episode success
        self._update_bullet_feedback(outcome)

        # 2. Reflect: Extract insights from episode (with multiple rounds)
        insights = self._reflect_on_episode_multi_round(
            episode=self.episode_history[-1],
            outcome=outcome,
            rounds=self.reflection_rounds
        )

        # 3. Curate: Organize insights into playbook updates
        delta_items = self._curate_insights(insights)

        # 4. Merge: Add to playbook
        self._merge_delta_items(delta_items)

        # 5. Deduplicate and prune: Remove redundancy and enforce token cap
        self._deduplicate_playbook()
        if self.token_cap:
            self._prune_playbook_to_cap()

        # Log outcome
        self.episode_history[-1]['outcome'] = outcome
        self.episode_history[-1]['insights'] = insights
        self.episode_history[-1]['delta_items'] = delta_items

        print(f"Playbook updated: {len(delta_items)} items added, "
              f"total size: {self._get_playbook_size()} bullets")

    # ========================================================================
    # Generator Methods (Action Selection)
    # ========================================================================

    def _choose_action(self, observation: dict) -> Tuple[str, Optional[str]]:
        """
        Decide next action using playbook (Generator).

        Args:
            observation: Current observation

        Returns:
            Tuple of (thought, action_string)
        """
        # Get tool descriptions
        if self.tools_class and hasattr(self.tools_class, 'get_tool_descriptions'):
            available_tools = self.tools_class.get_tool_descriptions()
        else:
            available_tools = "No tools available"

        # Format playbook as context (with retrieval if enabled)
        if self.use_retrieval and self.retriever:
            # Build query from observation and recent history
            query = f"{str(observation)} {format_observation_history(self.memory, max_steps=3)}"
            retrieved_playbook = self.retriever.retrieve_bullets(
                query=query,
                playbook=self.playbook,
                top_k=self.top_k,
                current_step=self.action_count
            )
            playbook_text = self._format_playbook(playbook=retrieved_playbook)
        else:
            playbook_text = self._format_playbook()

        # Build Generator prompt
        prompt = ACE_GENERATOR_TEMPLATE.format(
            playbook=playbook_text,
            observation=str(observation),
            memory_summary=format_observation_history(self.memory, max_steps=3),
            available_tools=available_tools,
            actions_remaining=self.action_budget - self.action_count
        )

        # Generate action
        response = self.llm.generate(prompt, temperature=self.generator_temperature)

        # Record token usage for exploration
        input_tokens, output_tokens = self.llm.get_last_usage()
        self.token_accountant.record(
            'exploration',
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            metadata={'action_count': self.action_count}
        )

        # Extract thought, action, and used bullets
        thought = extract_thought(response)
        action = extract_action(response)
        used_bullets = extract_used_bullets(response)

        # Track which bullets were referenced and update last_used_step
        if used_bullets:
            current_step = self.action_count
            for bullet_id in used_bullets:
                # Update last_used_step for each referenced bullet
                for section in self.playbook.values():
                    for bullet in section:
                        if bullet['id'] == bullet_id:
                            bullet['last_used_step'] = current_step
                            break

            # Track for feedback updates
            if self.episode_history:
                self.episode_history[-1]['referenced_bullets'].extend(used_bullets)

        return thought, action

    # ========================================================================
    # Reflector Methods (Insight Extraction)
    # ========================================================================

    def _reflect_on_episode_multi_round(self, episode: dict, outcome: dict,
                                        rounds: int = 1) -> Dict:
        """
        Reflect on episode with multiple rounds to strengthen insights.

        Args:
            episode: Episode dictionary with steps and outcome
            outcome: Test results and metrics
            rounds: Number of reflection rounds

        Returns:
            Final insights dictionary after all rounds
        """
        insights = None

        for round_num in range(rounds):
            # Reflect on episode (potentially incorporating prior round insights)
            if round_num == 0:
                insights = self._reflect_on_episode(episode, outcome)
            else:
                # Build on previous insights
                insights = self._reflect_on_episode(
                    episode, outcome, prior_insights=insights
                )

            # Record token usage for this round
            print(f"Reflection round {round_num + 1}/{rounds} completed")

        return insights

    def _reflect_on_episode(self, episode: dict, outcome: dict,
                           prior_insights: Optional[Dict] = None) -> Dict:
        """
        Reflector: Analyze episode to extract insights.

        Inputs:
        - Episode trajectory (observations, actions, outcomes)
        - Test query results (correctness, errors)
        - Prior insights from previous reflection round (if iterating)

        Outputs:
        - Reasoning about what worked/failed
        - Specific insights to add to playbook
        - Existing playbook items to mark helpful/harmful

        Args:
            episode: Episode dictionary with steps and outcome
            outcome: Test results and metrics
            prior_insights: Optional insights from previous round

        Returns:
            Dictionary with insights and reasoning
        """
        # Build reflection prompt
        if prior_insights:
            # Multi-round reflection: refine previous insights
            prior_text = f"\n\nPREVIOUS INSIGHTS (refine these):\n{json.dumps(prior_insights, indent=2)}"
        else:
            prior_text = ""

        prompt = ACE_REFLECTOR_TEMPLATE.format(
            playbook=self._format_playbook(),
            episode_steps=self._format_episode_steps(episode['steps']),
            test_results=json.dumps(outcome.get('test_results', {}), indent=2),
            environment_name=self.environment_name
        ) + prior_text

        # Query LLM for insights
        response = self.llm.generate(prompt, temperature=self.reflector_temperature)

        # Record token usage for planning (reflection)
        input_tokens, output_tokens = self.llm.get_last_usage()
        self.token_accountant.record(
            'planning',
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            metadata={'phase': 'reflection'}
        )

        # Parse insights
        try:
            insights = self._parse_json_response(response)
            return insights
        except Exception as e:
            print(f"Warning: Failed to parse reflector output: {e}")
            return {
                'what_worked': [],
                'what_failed': [],
                'new_insights': [],
                'helpful_bullets': [],
                'harmful_bullets': []
            }

    # ========================================================================
    # Curator Methods (Playbook Updates)
    # ========================================================================

    def _curate_insights(self, insights: Dict) -> List[Dict]:
        """
        Curator: Synthesize insights into compact delta items.

        Dispatches to different curation strategies based on self.curation_mode.

        Args:
            insights: Insights from Reflector

        Returns:
            List of delta items to add to playbook
        """
        # Dispatch to appropriate curation method
        if self.curation_mode == "curated":
            return self._curate_default(insights)
        elif self.curation_mode == "no_curate":
            return self._curate_append_only(insights)
        elif self.curation_mode == "random":
            return self._curate_random_subset(insights)
        elif self.curation_mode == "greedy":
            return self._curate_greedy_top_k(insights)
        else:
            print(f"Warning: Unknown curation mode '{self.curation_mode}', using default")
            return self._curate_default(insights)

    def _curate_default(self, insights: Dict) -> List[Dict]:
        """
        Default curated mode: LLM-based curation with deduplication.

        This is the standard ACE approach from the paper.

        Args:
            insights: Insights from Reflector

        Returns:
            List of delta items
        """
        # Build curation prompt
        prompt = ACE_CURATOR_TEMPLATE.format(
            playbook=self._format_playbook(),
            insights=json.dumps(insights, indent=2),
            environment_name=self.environment_name
        )

        # Query LLM for delta items
        response = self.llm.generate(prompt, temperature=self.curator_temperature)

        # Record token usage for curation
        input_tokens, output_tokens = self.llm.get_last_usage()
        self.token_accountant.record(
            'curation',
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            metadata={'phase': 'curation', 'mode': 'curated'}
        )

        # Parse delta items
        try:
            parsed = self._parse_json_response(response)
            delta_items = parsed.get('delta_items', [])

            # Enforce token cap if specified
            if self.token_cap:
                delta_items = self._enforce_token_cap(delta_items)

            return delta_items
        except Exception as e:
            print(f"Warning: Failed to parse curator output: {e}")
            return []

    def _curate_append_only(self, insights: Dict) -> List[Dict]:
        """
        No-curate mode: Append all insights without deduplication.

        Tests value of curation (H-Curation hypothesis).

        Args:
            insights: Insights from Reflector

        Returns:
            List of delta items (all insights appended)
        """
        delta_items = []

        # Extract all insights and convert to delta items
        for insight in insights.get('key_insights', []):
            delta_items.append({
                'section': 'strategies_and_hard_rules',
                'content': insight,
                'operation': 'add'
            })

        # Enforce token cap if specified
        if self.token_cap:
            delta_items = self._enforce_token_cap(delta_items)

        return delta_items

    def _curate_random_subset(self, insights: Dict) -> List[Dict]:
        """
        Random mode: Randomly select insights at same token budget.

        Tests whether selection matters or just having items matters.

        Args:
            insights: Insights from Reflector

        Returns:
            List of delta items (random subset)
        """
        import random

        # Extract all insights
        all_insights = insights.get('key_insights', [])

        # Randomly shuffle
        random.shuffle(all_insights)

        # Convert to delta items
        delta_items = [
            {
                'section': 'strategies_and_hard_rules',
                'content': insight,
                'operation': 'add'
            }
            for insight in all_insights
        ]

        # Enforce token cap
        if self.token_cap:
            delta_items = self._enforce_token_cap(delta_items)

        return delta_items

    def _curate_greedy_top_k(self, insights: Dict) -> List[Dict]:
        """
        Greedy mode: Select top K by helpfulness score only.

        Tests utility scoring without full curation.

        Args:
            insights: Insights from Reflector

        Returns:
            List of delta items (top K by helpfulness)
        """
        # Score insights by simple heuristic (length = detail)
        all_insights = insights.get('key_insights', [])

        # Score by length (longer = more detailed = potentially more helpful)
        scored = [(insight, len(insight)) for insight in all_insights]
        scored.sort(key=lambda x: x[1], reverse=True)

        # Take top insights
        delta_items = [
            {
                'section': 'strategies_and_hard_rules',
                'content': insight,
                'operation': 'add'
            }
            for insight, score in scored
        ]

        # Enforce token cap
        if self.token_cap:
            delta_items = self._enforce_token_cap(delta_items)

        return delta_items

    def _enforce_token_cap(self, delta_items: List[Dict]) -> List[Dict]:
        """
        Enforce token cap by truncating playbook to fit budget.

        Args:
            delta_items: New items to add

        Returns:
            Truncated list of delta items
        """
        # Estimate current playbook tokens
        current_tokens = self._estimate_playbook_tokens()

        # Add items until we hit the cap
        capped_items = []
        for item in delta_items:
            item_tokens = len(item['content'].split())  # Rough estimate
            if current_tokens + item_tokens <= self.token_cap:
                capped_items.append(item)
                current_tokens += item_tokens
            else:
                break

        return capped_items

    def _estimate_playbook_tokens(self) -> int:
        """
        Estimate current playbook token count.

        Returns:
            Estimated token count (rough)
        """
        playbook_text = self._format_playbook()
        # Rough estimate: 1 token ≈ 0.75 words
        word_count = len(playbook_text.split())
        return int(word_count / 0.75)

    def _merge_delta_items(self, delta_items: List[Dict]):
        """
        Merge delta items into playbook.

        Args:
            delta_items: List of items to add/update
        """
        for item in delta_items:
            section = item.get('section')
            operation = item.get('operation', 'add')

            if section not in self.playbook:
                print(f"Warning: Unknown section '{section}', skipping")
                continue

            if operation == 'add':
                self.playbook[section].append({
                    'id': self._generate_bullet_id(),
                    'content': item['content'],
                    'helpful_count': 0,
                    'harmful_count': 0,
                    'last_used_step': None
                })
            elif operation == 'update':
                # Update existing bullet (if bullet_id provided)
                bullet_id = item.get('bullet_id')
                if bullet_id:
                    self._update_bullet(section, bullet_id, item.get('updates', {}))

    def _update_bullet(self, section: str, bullet_id: str, updates: dict):
        """
        Update existing bullet in playbook.

        Args:
            section: Playbook section name
            bullet_id: Bullet ID to update
            updates: Dictionary of fields to update
        """
        for bullet in self.playbook[section]:
            if bullet['id'] == bullet_id:
                bullet.update(updates)
                break

    # ========================================================================
    # Bullet Feedback Methods
    # ========================================================================

    def _update_bullet_feedback(self, outcome: dict):
        """
        Update helpful_count/harmful_count for bullets used in this episode.

        Args:
            outcome: Episode outcome with test results
        """
        if not self.episode_history:
            return

        # Get referenced bullets from this episode
        referenced_bullets = self.episode_history[-1].get('referenced_bullets', [])

        # Determine if episode was successful
        # test_results is a list of test result dicts from runner.py
        test_results = outcome.get('test_results', [])
        # Episode is successful if all test queries were correct
        is_success = all(result.get('correct', False) for result in test_results) if test_results else False

        # Update counts for all referenced bullets
        for bullet_id in set(referenced_bullets):  # Deduplicate
            for section in self.playbook.values():
                for bullet in section:
                    if bullet['id'] == bullet_id:
                        if is_success:
                            bullet['helpful_count'] += 1
                        else:
                            bullet['harmful_count'] += 1
                        break

    # ========================================================================
    # Deduplication and Pruning Methods
    # ========================================================================

    def _deduplicate_playbook(self):
        """
        Remove redundant bullets using semantic similarity.

        Uses simple text-based similarity for now. Could be upgraded to
        embedding-based similarity for better accuracy.
        """
        try:
            from difflib import SequenceMatcher
        except ImportError:
            print("Warning: difflib not available, skipping deduplication")
            return

        for section_name in self.playbook:
            bullets = self.playbook[section_name]
            if len(bullets) <= 1:
                continue

            # Track bullets to keep
            to_keep = []
            for i, bullet in enumerate(bullets):
                # Check if this bullet is similar to any we're keeping
                is_duplicate = False
                for kept_bullet in to_keep:
                    similarity = SequenceMatcher(
                        None,
                        bullet['content'].lower(),
                        kept_bullet['content'].lower()
                    ).ratio()

                    # If very similar (>80% match), consider it a duplicate
                    if similarity > 0.8:
                        is_duplicate = True
                        # Merge feedback counts into the kept bullet
                        kept_bullet['helpful_count'] += bullet['helpful_count']
                        kept_bullet['harmful_count'] += bullet['harmful_count']
                        break

                if not is_duplicate:
                    to_keep.append(bullet)

            # Update section with deduplicated bullets
            self.playbook[section_name] = to_keep

    def _prune_playbook_to_cap(self):
        """
        Prune playbook to fit within token cap using utility scoring.

        Keeps bullets with highest utility score (helpful - harmful).
        """
        if not self.token_cap:
            return

        # Get all bullets with utility scores
        all_bullets = []
        for section_name, bullets in self.playbook.items():
            for bullet in bullets:
                utility = bullet['helpful_count'] - bullet['harmful_count']
                all_bullets.append({
                    'section': section_name,
                    'bullet': bullet,
                    'utility': utility
                })

        # Sort by utility (descending)
        all_bullets.sort(key=lambda x: x['utility'], reverse=True)

        # Rebuild playbook within token cap
        new_playbook = self._init_playbook()
        current_tokens = 0

        for item in all_bullets:
            bullet_tokens = len(item['bullet']['content'].split())
            if current_tokens + bullet_tokens <= self.token_cap:
                new_playbook[item['section']].append(item['bullet'])
                current_tokens += bullet_tokens
            else:
                break

        self.playbook = new_playbook

    # ========================================================================
    # Playbook Management & Formatting
    # ========================================================================

    def _format_playbook(self, playbook: Optional[Dict] = None) -> str:
        """
        Format playbook as text for prompts.

        Args:
            playbook: Optional playbook to format (defaults to self.playbook)

        Returns:
            Formatted playbook string
        """
        pb = playbook if playbook is not None else self.playbook

        # Check if playbook is empty
        total_bullets = sum(len(bullets) for bullets in pb.values())
        if total_bullets == 0:
            return "**PLAYBOOK EMPTY** - No strategies learned yet."

        sections = []
        for section_name, bullets in pb.items():
            if bullets:
                section_text = f"\n## {section_name.replace('_', ' ').upper()}\n"
                for i, bullet in enumerate(bullets):
                    # Include bullet ID, content, and utility score
                    bullet_id = bullet.get('id', 'unknown')
                    utility = bullet.get('helpful_count', 0) - bullet.get('harmful_count', 0)
                    utility_str = f" [utility:{utility:+d}]" if utility != 0 else ""
                    section_text += f"{i+1}. [ID:{bullet_id}] {bullet['content']}{utility_str}\n"
                sections.append(section_text)

        return '\n'.join(sections) if sections else "**PLAYBOOK EMPTY**"

    def _format_episode_steps(self, steps: List[dict]) -> str:
        """
        Format episode steps for Reflector prompt.

        Args:
            steps: List of step dictionaries

        Returns:
            Formatted string
        """
        lines = []
        for i, step in enumerate(steps):
            action = step.get('action', 'observe')
            obs = str(step.get('observation', {}))[:100]
            lines.append(f"Step {i}: {action} -> {obs}")

        return '\n'.join(lines) if lines else "No steps recorded"

    def _get_playbook_size(self) -> int:
        """
        Get total number of bullets in playbook.

        Returns:
            Total bullet count
        """
        return sum(len(bullets) for bullets in self.playbook.values())

    def _get_playbook_snapshot(self) -> dict:
        """
        Get snapshot of current playbook state.

        Returns:
            Dictionary with playbook metadata
        """
        return {
            'total_bullets': self._get_playbook_size(),
            'sections': {
                section: len(bullets)
                for section, bullets in self.playbook.items()
            }
        }

    def _flatten_playbook(self) -> List[Dict]:
        """
        Flatten playbook to list of all bullets.

        Returns:
            List of all bullets across sections
        """
        all_bullets = []
        for section_name, bullets in self.playbook.items():
            for bullet in bullets:
                all_bullets.append({
                    'section': section_name,
                    **bullet
                })
        return all_bullets

    def _generate_bullet_id(self) -> str:
        """
        Generate unique bullet ID.

        Returns:
            UUID string
        """
        return str(uuid.uuid4())[:8]

    # ========================================================================
    # Token Accounting
    # ========================================================================

    def get_token_breakdown(self) -> dict:
        """
        Get token breakdown by category.

        Returns:
            Dictionary with token breakdown and validation status
        """
        return self.token_accountant.to_dict()

    def validate_token_accounting(self, total_input: int, total_output: int) -> bool:
        """
        Validate that token breakdown matches totals.

        Args:
            total_input: Expected total input tokens
            total_output: Expected total output tokens

        Returns:
            True if validation passes

        Raises:
            ValueError: If validation fails
        """
        return self.token_accountant.validate(total_input, total_output)

    # ========================================================================
    # JSON Parsing Helpers
    # ========================================================================

    def _parse_json_response(self, response: str) -> dict:
        """
        Parse JSON from LLM response with robust error handling.

        Args:
            response: Raw LLM response

        Returns:
            Parsed JSON dictionary

        Raises:
            ValueError: If JSON cannot be parsed
        """
        # Try 1: Direct JSON parse
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass

        # Try 2: Extract JSON from markdown code blocks
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # Try 3: Find first {...} block
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass

        # All parsing failed
        raise ValueError(f"Could not parse JSON from response: {response[:200]}")
