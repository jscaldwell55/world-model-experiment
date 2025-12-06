"""
Training Data Generation for World Model Graduation

Converts playbook observations into instruction/response training pairs
suitable for LoRA fine-tuning.

Each domain has specific knowledge types:
- HotPot: heating rates, temperature dynamics, power settings
- ChemTile: chemical reactions, mixing rules, outcomes
- SwitchLight: switch-light mappings, wiring layouts
"""

import json
import hashlib
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field, asdict
from collections import defaultdict


@dataclass
class TrainingPair:
    """A single instruction/response training pair"""
    instruction: str
    response: str
    domain: str
    source_episode: str
    reliability: str
    score: float
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return asdict(self)

    def content_hash(self) -> str:
        """Hash for deduplication based on instruction+response content"""
        content = f"{self.instruction}|{self.response}"
        return hashlib.md5(content.encode()).hexdigest()[:12]


class TrainingDataGenerator:
    """Generate training pairs from playbook observations"""

    def __init__(self, playbook_base: str = "memory/domains"):
        self.playbook_base = Path(playbook_base)
        self.domains = ['hot_pot', 'chem_tile', 'switch_light']

    def load_playbook(self, domain: str) -> Dict:
        """Load playbook for a domain"""
        playbook_path = self.playbook_base / domain / 'playbook.json'
        if not playbook_path.exists():
            return {'observations': []}
        with open(playbook_path) as f:
            return json.load(f)

    def filter_by_reliability(
        self,
        observations: List[Dict],
        min_reliability: str = 'HIGH'
    ) -> List[Dict]:
        """
        Filter observations by reliability threshold.

        Args:
            observations: List of observation dicts
            min_reliability: Minimum reliability level
                'HIGH' - only HIGH and SYNTHETIC_HIGH
                'MEDIUM' - HIGH, MEDIUM, SYNTHETIC_HIGH, SYNTHETIC_MEDIUM
                'LOW' - all observations

        Returns:
            Filtered list of observations
        """
        if min_reliability == 'LOW':
            return observations

        high_tags = ['HIGH', 'SYNTHETIC_HIGH']
        medium_tags = ['MEDIUM', 'SYNTHETIC_MEDIUM']

        if min_reliability == 'HIGH':
            valid_tags = high_tags
        elif min_reliability == 'MEDIUM':
            valid_tags = high_tags + medium_tags
        else:
            valid_tags = high_tags

        return [obs for obs in observations if obs.get('reliability') in valid_tags]

    def generate_pairs_hot_pot(self, observation: Dict) -> List[TrainingPair]:
        """
        Generate training pairs for HotPot domain.

        Knowledge types:
        - Heating rates at different power settings
        - Temperature dynamics
        - Power setting effects
        """
        pairs = []
        beliefs = observation.get('beliefs', {})
        context = observation.get('context', {})
        episode_id = observation.get('episode_id', 'unknown')
        reliability = observation.get('reliability', 'UNKNOWN')
        score = observation.get('score', 0.0)

        # Extract values
        heating_rate = beliefs.get('heating_rate_mean', {}).get('value')
        base_temp = beliefs.get('base_temp', {}).get('value', 20.0)
        power_setting = context.get('power_setting', 'UNKNOWN')

        if heating_rate is None:
            return pairs

        # Pair 1: Direct heating rate question
        pairs.append(TrainingPair(
            instruction=f"What is the heating rate when the stove is set to {power_setting} power?",
            response=f"At {power_setting} power, the heating rate is approximately {heating_rate:.1f}°C per second.",
            domain='hot_pot',
            source_episode=episode_id,
            reliability=reliability,
            score=score,
            metadata={'power_setting': power_setting, 'heating_rate': heating_rate}
        ))

        # Pair 2: Temperature prediction question
        if heating_rate > 0:
            time_to_boil = (100 - base_temp) / heating_rate
            pairs.append(TrainingPair(
                instruction=f"How long will it take to heat water from {base_temp:.0f}°C to boiling at {power_setting} power?",
                response=f"At {power_setting} power with a heating rate of {heating_rate:.1f}°C/s, it will take approximately {time_to_boil:.0f} seconds to reach 100°C from {base_temp:.0f}°C.",
                domain='hot_pot',
                source_episode=episode_id,
                reliability=reliability,
                score=score,
                metadata={'power_setting': power_setting, 'time_to_boil': time_to_boil}
            ))

        # Pair 3: Power comparison (if HIGH power)
        if power_setting == 'HIGH' and heating_rate > 1.5:
            pairs.append(TrainingPair(
                instruction="Which power setting heats water faster, HIGH or LOW?",
                response=f"HIGH power heats water faster at {heating_rate:.1f}°C/s. LOW power typically heats at around 1.0°C/s.",
                domain='hot_pot',
                source_episode=episode_id,
                reliability=reliability,
                score=score,
                metadata={'comparison': True}
            ))

        # Pair 4: Temperature at specific time
        if heating_rate > 0:
            temp_at_30s = base_temp + heating_rate * 30
            pairs.append(TrainingPair(
                instruction=f"What temperature will the pot reach after 30 seconds at {power_setting} power?",
                response=f"Starting from {base_temp:.0f}°C at {power_setting} power ({heating_rate:.1f}°C/s), the pot will reach approximately {temp_at_30s:.0f}°C after 30 seconds.",
                domain='hot_pot',
                source_episode=episode_id,
                reliability=reliability,
                score=score,
                metadata={'time': 30, 'predicted_temp': temp_at_30s}
            ))

        return pairs

    def generate_pairs_chem_tile(self, observation: Dict) -> List[TrainingPair]:
        """
        Generate training pairs for ChemTile domain.

        Knowledge types:
        - Chemical reaction outcomes
        - Mixing rules
        - Temperature effects
        """
        pairs = []
        beliefs = observation.get('beliefs', {})
        episode_id = observation.get('episode_id', 'unknown')
        reliability = observation.get('reliability', 'UNKNOWN')
        score = observation.get('score', 0.0)

        # Extract reaction probabilities
        reaction_probs = beliefs.get('reaction_probs', {}).get('value', {})
        temperature = beliefs.get('temperature', {}).get('value', 'medium')

        if not reaction_probs:
            return pairs

        # Generate pairs for each reaction
        for reaction, outcomes in reaction_probs.items():
            if not isinstance(outcomes, dict):
                continue

            # Find most likely outcome
            best_outcome = max(outcomes.items(), key=lambda x: x[1])
            outcome_name, prob = best_outcome

            # Parse reaction (e.g., "A+B" -> compounds A and B)
            if '+' in reaction:
                compounds = reaction.split('+')
                compound_a = compounds[0].strip()
                compound_b = compounds[1].strip()
            else:
                continue

            # Pair 1: What happens when mixing
            pairs.append(TrainingPair(
                instruction=f"What happens when you mix compound {compound_a} with compound {compound_b}?",
                response=f"Mixing {compound_a} with {compound_b} most likely produces {outcome_name} (probability: {prob:.0%}). Other possible outcomes include: {', '.join(f'{k} ({v:.0%})' for k, v in outcomes.items() if k != outcome_name)}.",
                domain='chem_tile',
                source_episode=episode_id,
                reliability=reliability,
                score=score,
                metadata={'reaction': reaction, 'outcome': outcome_name, 'probability': prob}
            ))

            # Pair 2: How to create a specific compound
            if outcome_name not in ['explode', 'nothing']:
                pairs.append(TrainingPair(
                    instruction=f"How do I create compound {outcome_name}?",
                    response=f"To create compound {outcome_name}, mix {compound_a} and {compound_b}. This reaction has a {prob:.0%} success rate at {temperature} temperature.",
                    domain='chem_tile',
                    source_episode=episode_id,
                    reliability=reliability,
                    score=score,
                    metadata={'target_compound': outcome_name}
                ))

            # Pair 3: Safety warning if explosion risk
            if 'explode' in outcomes and outcomes['explode'] > 0.05:
                explode_prob = outcomes['explode']
                pairs.append(TrainingPair(
                    instruction=f"Is it safe to mix {compound_a} and {compound_b}?",
                    response=f"Mixing {compound_a} and {compound_b} has a {explode_prob:.0%} chance of explosion. Exercise caution and ensure proper safety measures.",
                    domain='chem_tile',
                    source_episode=episode_id,
                    reliability=reliability,
                    score=score,
                    metadata={'safety_warning': True, 'explosion_risk': explode_prob}
                ))

        # Temperature effects
        if temperature:
            pairs.append(TrainingPair(
                instruction="What temperature setting is best for chemical reactions in this lab?",
                response=f"The recommended temperature setting is {temperature}. This provides optimal conditions for most reactions while minimizing explosion risk.",
                domain='chem_tile',
                source_episode=episode_id,
                reliability=reliability,
                score=score,
                metadata={'temperature': temperature}
            ))

        return pairs

    def generate_pairs_switch_light(self, observation: Dict) -> List[TrainingPair]:
        """
        Generate training pairs for SwitchLight domain.

        Knowledge types:
        - Wiring layout probabilities
        - Switch-light mappings
        - Diagnostic procedures
        """
        pairs = []
        beliefs = observation.get('beliefs', {})
        episode_id = observation.get('episode_id', 'unknown')
        reliability = observation.get('reliability', 'UNKNOWN')
        score = observation.get('score', 0.0)

        # Extract wiring probabilities
        wiring_probs = beliefs.get('wiring_probs', {}).get('value', {})
        failure_prob = beliefs.get('failure_prob', {}).get('value', 0.02)

        if not wiring_probs:
            return pairs

        # Find most likely layout
        if wiring_probs:
            best_layout = max(wiring_probs.items(), key=lambda x: x[1])
            layout_name, prob = best_layout

            # Pair 1: Most likely wiring layout
            pairs.append(TrainingPair(
                instruction="What wiring layout is most likely in this building?",
                response=f"Based on observations, {layout_name} is the most likely wiring layout with {prob:.0%} probability. Other layouts: {', '.join(f'{k} ({v:.0%})' for k, v in wiring_probs.items() if k != layout_name)}.",
                domain='switch_light',
                source_episode=episode_id,
                reliability=reliability,
                score=score,
                metadata={'layout': layout_name, 'probability': prob}
            ))

            # Pair 2: How to identify wiring
            pairs.append(TrainingPair(
                instruction="How can I determine the wiring layout?",
                response=f"To determine the wiring layout, systematically flip switches and observe which lights respond. The current evidence suggests {layout_name} layout ({prob:.0%} confidence). Multiple switch tests increase certainty.",
                domain='switch_light',
                source_episode=episode_id,
                reliability=reliability,
                score=score,
                metadata={'diagnostic': True}
            ))

        # Pair 3: Failure probability
        if failure_prob is not None:
            pairs.append(TrainingPair(
                instruction="How reliable is the switch-light system?",
                response=f"The system has approximately {failure_prob:.1%} failure probability. Most switch operations will work correctly, but occasional failures should be expected.",
                domain='switch_light',
                source_episode=episode_id,
                reliability=reliability,
                score=score,
                metadata={'failure_prob': failure_prob}
            ))

        # Pair 4: Troubleshooting
        pairs.append(TrainingPair(
            instruction="What should I do if a light doesn't respond to a switch?",
            response=f"If a light doesn't respond: 1) Try flipping the switch again (failure rate is {failure_prob:.1%}). 2) Check if the wiring layout matches {list(wiring_probs.keys())[0] if wiring_probs else 'expected configuration'}. 3) Inspect for relay issues.",
            domain='switch_light',
            source_episode=episode_id,
            reliability=reliability,
            score=score,
            metadata={'troubleshooting': True}
        ))

        return pairs

    def generate_all_pairs(
        self,
        min_reliability: str = 'HIGH',
        deduplicate: bool = True
    ) -> Tuple[List[TrainingPair], Dict]:
        """
        Generate training pairs from all domains.

        Args:
            min_reliability: Minimum reliability level ('HIGH', 'MEDIUM', 'LOW')
            deduplicate: If True, remove duplicate instruction/response pairs

        Returns:
            Tuple of (list of TrainingPairs, statistics dict)
        """
        all_pairs = []
        stats = {
            'total_observations': 0,
            'filtered_observations': 0,
            'pairs_generated': 0,
            'pairs_after_dedup': 0,
            'by_domain': {}
        }

        generators = {
            'hot_pot': self.generate_pairs_hot_pot,
            'chem_tile': self.generate_pairs_chem_tile,
            'switch_light': self.generate_pairs_switch_light
        }

        for domain in self.domains:
            playbook = self.load_playbook(domain)
            observations = playbook.get('observations', [])

            stats['total_observations'] += len(observations)

            # Filter by reliability
            filtered = self.filter_by_reliability(observations, min_reliability)
            stats['filtered_observations'] += len(filtered)

            # Generate pairs
            domain_pairs = []
            generator = generators.get(domain)

            if generator:
                for obs in filtered:
                    pairs = generator(obs)
                    domain_pairs.extend(pairs)

            stats['by_domain'][domain] = {
                'observations': len(observations),
                'filtered': len(filtered),
                'pairs': len(domain_pairs)
            }

            all_pairs.extend(domain_pairs)

        stats['pairs_generated'] = len(all_pairs)

        # Deduplicate
        if deduplicate:
            seen_hashes = set()
            unique_pairs = []
            for pair in all_pairs:
                h = pair.content_hash()
                if h not in seen_hashes:
                    seen_hashes.add(h)
                    unique_pairs.append(pair)
            all_pairs = unique_pairs

        stats['pairs_after_dedup'] = len(all_pairs)

        return all_pairs, stats

    def save_pairs(
        self,
        pairs: List[TrainingPair],
        output_path: str,
        format: str = 'json'
    ):
        """
        Save training pairs to file.

        Args:
            pairs: List of TrainingPair objects
            output_path: Output file path
            format: 'json' or 'jsonl'
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == 'json':
            with open(output_path, 'w') as f:
                json.dump([p.to_dict() for p in pairs], f, indent=2)
        elif format == 'jsonl':
            with open(output_path, 'w') as f:
                for pair in pairs:
                    f.write(json.dumps(pair.to_dict()) + '\n')

    def load_pairs(self, input_path: str) -> List[TrainingPair]:
        """Load training pairs from file."""
        input_path = Path(input_path)

        if input_path.suffix == '.jsonl':
            pairs = []
            with open(input_path) as f:
                for line in f:
                    data = json.loads(line)
                    pairs.append(TrainingPair(**data))
            return pairs
        else:
            with open(input_path) as f:
                data = json.load(f)
            return [TrainingPair(**p) for p in data]


def generate_training_pairs(
    min_reliability: str = 'HIGH',
    deduplicate: bool = True,
    playbook_base: str = "memory/domains"
) -> Tuple[List[Dict], Dict]:
    """
    Convenience function to generate training pairs.

    Returns:
        Tuple of (list of pair dicts, statistics dict)
    """
    generator = TrainingDataGenerator(playbook_base)
    pairs, stats = generator.generate_all_pairs(min_reliability, deduplicate)
    return [p.to_dict() for p in pairs], stats
