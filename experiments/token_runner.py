"""Episode runner with parallel token prediction logging.

This module extends the existing experiment pipeline to add token-level
prediction capability without modifying core agent/environment logic.
"""

from typing import Dict, List, Tuple, Optional
import os
from pathlib import Path

from textualization.base import TextualizationLayer
from token_prediction.predictor import NextSentencePredictor
from token_prediction.logger import TokenLogger, TokenLogEntry


def run_episode_with_tokens(
    env,
    agent,
    textualizer: TextualizationLayer,
    predictor: NextSentencePredictor,
    seed: int,
    max_actions: int = 10,
    save_dir: Optional[str] = None,
    control_mode: Optional[str] = None
) -> Tuple[Dict, TokenLogger]:
    """
    Run episode with parallel token prediction logging.

    This function runs a normal episode while simultaneously:
    1. Textualizing observations into canonical strings
    2. Querying LLM for next-observation predictions (with logprobs)
    3. Logging token NLL alongside belief surprisal

    The core agent/environment loop is unchanged - token prediction
    happens in parallel and doesn't affect the episode execution.

    Args:
        env: Environment instance (HotPotLab, SwitchLight, ChemTile)
        agent: Agent instance (Observer, Actor, etc.)
        textualizer: Textualization layer for environment
        predictor: Token-level predictor (OpenAI, etc.)
        seed: Random seed for episode
        max_actions: Maximum actions per episode
        save_dir: Directory to save token logs
        control_mode: Optional negative control ('shuffled', 'random', or None)
                     If set, wraps textualizer to break semantic coupling

    Returns:
        (test_results, token_logger): Test query results and token log

    Raises:
        ValueError: If environment and textualizer are incompatible
        ValueError: If control_mode is invalid
    """

    # Wrap textualizer with negative control if requested
    if control_mode is not None:
        from textualization.negative_controls import create_negative_control_textualizer
        textualizer = create_negative_control_textualizer(
            textualizer,
            control_type=control_mode,
            seed=seed
        )
        control_suffix = f"_ctrl_{control_mode}"
    else:
        control_suffix = ""

    # Initialize
    env_name = env.__class__.__name__
    agent_name = agent.__class__.__name__
    episode_id = f"{env_name}_{agent_name}_ep{seed:03d}{control_suffix}"
    token_logger = TokenLogger(episode_id)

    # Build initial context
    obs = env.reset(seed)
    initial_obs_text = textualizer.textualize_observation(obs)
    context_history = [initial_obs_text]

    print(f"  Starting episode: {episode_id}")
    if control_mode:
        print(f"  ⚠️  NEGATIVE CONTROL MODE: {control_mode}")
    print(f"  Initial observation: {initial_obs_text[:60]}...")

    # Episode loop
    for step in range(max_actions):
        # Get agent action (existing pipeline - no modifications)
        agent_step = agent.act(obs)
        action = agent_step.action

        if action is None or action == 'done':
            print(f"  Episode ended at step {step} (action={action})")
            break

        # Strip empty parentheses from actions (for environments that don't parse them)
        # e.g., "measure_temp()" → "measure_temp"
        # but keep "wait(5)" → "wait(5)"
        if action and action.endswith("()"):
            action = action[:-2]

        # === TOKEN PREDICTION PARALLEL TRACK ===

        # Textualize action
        action_text = textualizer.textualize_action(action)

        # Build full context for prediction
        context = "\n".join(context_history + [action_text])

        # Predict next observation (token-level)
        token_pred = None
        try:
            token_pred = predictor.predict_next_observation(
                context,
                temperature=0.0,
                max_tokens=100
            )
        except Exception as e:
            print(f"  Warning: Token prediction failed at step {step}: {e}")
            # Continue episode even if API fails

        # === EXECUTE ACTION (existing pipeline) ===
        next_obs, reward, done, info = env.step(action)

        # Textualize true observation
        true_obs_text = textualizer.textualize_observation(next_obs)

        # === EXTRACT BELIEF SURPRISAL (from existing pipeline) ===
        belief_surprisal = None

        # For Actor agents, compute surprisal using the agent's method
        if hasattr(agent, 'compute_surprisal'):
            try:
                # Compute surprisal with CURRENT belief (before update)
                belief_surprisal = agent.compute_surprisal(next_obs)

                # Update belief with new observation for next iteration
                if hasattr(agent, 'update_belief_from_observation'):
                    agent.update_belief_from_observation(next_obs)
            except Exception as e:
                print(f"  Warning: Belief surprisal computation failed: {e}")

        # === LOG TOKEN DATA ===
        if token_pred is not None:
            token_logger.log_step(TokenLogEntry(
                step=step,
                context_text=context,
                true_observation=true_obs_text,
                predicted_text=token_pred.predicted_text,
                tokens=token_pred.tokens,
                logprobs=token_pred.logprobs,
                sequence_nll=token_pred.sequence_nll,
                per_token_nll=token_pred.per_token_nll,
                belief_surprisal=belief_surprisal,
                accuracy=None  # Will fill after test queries
            ))

            surprisal_str = f"{belief_surprisal:.2f}" if belief_surprisal is not None else "N/A"
            print(f"  Step {step}: NLL={token_pred.sequence_nll:.2f}, Belief Surprisal={surprisal_str}")

        # Update context history
        context_history.append(action_text)
        context_history.append(true_obs_text)

        # Continue episode
        obs = next_obs
        if done:
            print(f"  Episode done at step {step}")
            break

    # Run test queries (existing pipeline)
    test_results = {}
    if hasattr(agent, 'answer_test_queries'):
        try:
            test_results = agent.answer_test_queries()
            print(f"  Test queries completed: {len(test_results)} queries")
        except Exception as e:
            print(f"  Warning: Test queries failed: {e}")

    # Compute average accuracy and add to token logs
    if isinstance(test_results, list) and len(test_results) > 0:
        avg_accuracy = sum(r.get('correct', 0) for r in test_results) / len(test_results)
        for entry in token_logger.entries:
            entry.accuracy = avg_accuracy
        print(f"  Average accuracy: {avg_accuracy:.2%}")

    # Save token log
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        log_path = os.path.join(save_dir, f"{episode_id}_token.json")
        token_logger.save(log_path)
        print(f"  Saved token log to: {log_path}")

    return test_results, token_logger


def create_textualizer(env) -> TextualizationLayer:
    """Factory function to create appropriate textualizer for environment.

    Args:
        env: Environment instance

    Returns:
        TextualizationLayer instance matching the environment

    Raises:
        ValueError: If no textualizer exists for environment type
    """
    from textualization import (
        HotPotTextualization,
        SwitchLightTextualization,
        ChemTileTextualization
    )

    env_name = env.__class__.__name__

    if 'HotPot' in env_name:
        return HotPotTextualization()
    elif 'SwitchLight' in env_name:
        return SwitchLightTextualization()
    elif 'ChemTile' in env_name or 'Chem' in env_name:
        return ChemTileTextualization()
    else:
        raise ValueError(f"No textualizer for environment: {env_name}")


def create_predictor(model: str = "gpt-4o-mini", provider: str = "openai") -> NextSentencePredictor:
    """Factory function to create token predictor.

    Args:
        model: Model name (e.g., "gpt-4o-mini", "gpt-4o")
        provider: Provider name (currently only "openai" supported)

    Returns:
        NextSentencePredictor instance

    Raises:
        ValueError: If provider is not supported
        ImportError: If required packages not installed
    """
    if provider == "openai":
        from token_prediction.openai_predictor import OpenAINextSentencePredictor
        return OpenAINextSentencePredictor(model=model)
    else:
        raise ValueError(f"Unsupported predictor provider: {provider}")
