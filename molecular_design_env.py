"""
MolecularDesignEnv: Agentic environment for molecular property exploration.

This environment simulates a realistic drug discovery scenario where an agent
must decide which molecules to query from a candidate pool with a limited budget.
The world model helps decide what to query next based on predictions and uncertainty.

Key insight: This creates TRAJECTORIES for Offline Consolidation (OC), not single-step
predictions. Each episode is a sequence of decisions that OC can learn from.
"""

import copy
import pickle
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd

from molecular_world_model import MolecularWorldModel, load_esol_data


# ============================================================================
# DATA CORRUPTION FOR STRESS TESTING
# ============================================================================

class DataCorruptor:
    """
    Inject realistic data quality issues for stress testing OC.

    Modes:
    - 'none': No corruption (clean data)
    - 'noise': Random label noise (percentage of labels get ±noise_magnitude)
    - 'shift': Distribution shift - only allow certain contexts in training
    - 'imbalance': Oversample certain scaffold clusters
    - 'drift': Temporal drift - labels shift after a threshold step
    """

    def __init__(self, mode: str = 'none', severity: float = 0.15,
                 noise_magnitude: float = 0.5, drift_offset: float = 0.3,
                 drift_threshold: int = 25, seed: int = 42):
        """
        Initialize DataCorruptor.

        Args:
            mode: Corruption mode ('none', 'noise', 'shift', 'imbalance', 'drift')
            severity: Corruption severity (fraction affected for noise/imbalance)
            noise_magnitude: Magnitude of noise to add for 'noise' mode
            drift_offset: Label offset for 'drift' mode
            drift_threshold: Step after which drift occurs
            seed: Random seed for reproducibility
        """
        self.mode = mode
        self.severity = severity
        self.noise_magnitude = noise_magnitude
        self.drift_offset = drift_offset
        self.drift_threshold = drift_threshold
        self.seed = seed
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)

        # Track corruption statistics
        self.n_corrupted = 0
        self.n_total = 0
        self.corruption_log = []

    def reset(self):
        """Reset corruption statistics."""
        self.n_corrupted = 0
        self.n_total = 0
        self.corruption_log = []
        self.rng = random.Random(self.seed)
        self.np_rng = np.random.RandomState(self.seed)

    def corrupt_label(self, true_label: float, step: int = 0,
                      context: Optional[Tuple[int, int, int]] = None) -> float:
        """
        Corrupt a label based on the corruption mode.

        Args:
            true_label: The true property value
            step: Current step in the episode
            context: (scaffold_cluster, mw_bin, logp_bin) tuple

        Returns:
            Corrupted label (may be same as true_label)
        """
        self.n_total += 1

        if self.mode == 'none':
            return true_label

        elif self.mode == 'noise':
            # Percentage of labels get random noise
            if self.rng.random() < self.severity:
                noise = self.rng.choice([-self.noise_magnitude, self.noise_magnitude])
                self.n_corrupted += 1
                self.corruption_log.append({
                    'step': step,
                    'type': 'noise',
                    'original': true_label,
                    'corrupted': true_label + noise,
                    'noise': noise
                })
                return true_label + noise
            return true_label

        elif self.mode == 'drift':
            # After threshold step, labels systematically shift
            if step > self.drift_threshold:
                self.n_corrupted += 1
                self.corruption_log.append({
                    'step': step,
                    'type': 'drift',
                    'original': true_label,
                    'corrupted': true_label + self.drift_offset,
                    'offset': self.drift_offset
                })
                return true_label + self.drift_offset
            return true_label

        elif self.mode == 'shift':
            # For shift mode, corruption is done at filtering level, not label level
            return true_label

        elif self.mode == 'imbalance':
            # For imbalance mode, corruption is done at filtering level
            return true_label

        else:
            return true_label

    def filter_candidates(self, candidate_indices: List[int],
                          contexts: List[Tuple[int, int, int]],
                          descriptors: Optional[List[Dict]] = None) -> List[int]:
        """
        Filter candidate pool based on corruption mode.

        Used for 'shift' and 'imbalance' modes to create biased candidate pools.

        Args:
            candidate_indices: List of candidate indices
            contexts: Context tuples for each candidate
            descriptors: Optional descriptors dict for each candidate

        Returns:
            Filtered list of candidate indices
        """
        if self.mode == 'none' or self.mode == 'noise' or self.mode == 'drift':
            return candidate_indices

        elif self.mode == 'shift':
            # Only allow low-MW molecules (mw_bin < 2)
            filtered = []
            for idx, ctx in zip(candidate_indices, contexts):
                mw_bin = ctx[1]  # (scaffold, mw_bin, logp_bin)
                if mw_bin < 2:
                    filtered.append(idx)

            # If too restrictive, allow some random samples
            if len(filtered) < len(candidate_indices) * 0.2:
                additional_needed = int(len(candidate_indices) * 0.2) - len(filtered)
                remaining = [i for i in candidate_indices if i not in filtered]
                if remaining:
                    additional = self.np_rng.choice(
                        remaining,
                        size=min(additional_needed, len(remaining)),
                        replace=False
                    ).tolist()
                    filtered.extend(additional)

            return filtered

        elif self.mode == 'imbalance':
            # Oversample scaffold cluster 0 (severity controls the imbalance ratio)
            cluster_0 = []
            others = []

            for idx, ctx in zip(candidate_indices, contexts):
                scaffold_cluster = ctx[0]
                if scaffold_cluster == 0:
                    cluster_0.append(idx)
                else:
                    others.append(idx)

            # Oversample cluster 0 based on severity
            # severity=0.5 means cluster 0 appears 2x more often
            if cluster_0 and others:
                oversample_factor = 1 / (1 - self.severity) if self.severity < 1 else 3
                n_cluster_0_copies = int(len(cluster_0) * oversample_factor)

                # Create oversampled list
                oversampled_cluster_0 = (cluster_0 * int(np.ceil(n_cluster_0_copies / len(cluster_0))))[:n_cluster_0_copies]

                return oversampled_cluster_0 + others

            return candidate_indices

        return candidate_indices

    def get_stats(self) -> Dict:
        """Get corruption statistics."""
        return {
            'mode': self.mode,
            'severity': self.severity,
            'n_corrupted': self.n_corrupted,
            'n_total': self.n_total,
            'corruption_rate': self.n_corrupted / self.n_total if self.n_total > 0 else 0,
            'n_log_entries': len(self.corruption_log)
        }


@dataclass
class StepRecord:
    """Record of a single environment step."""
    step: int
    action: int  # Index in candidate pool
    smiles: str
    context: Tuple[int, int, int]
    prediction: float
    uncertainty: float
    true_label: float
    error: float
    reward: float
    cumulative_reward: float
    remaining_budget: int
    pool_mean_uncertainty: float  # Average uncertainty across unqueried pool


@dataclass
class Episode:
    """Complete episode trajectory for OC consumption."""
    env_name: str = 'molecular_design'
    steps: List[Dict] = field(default_factory=list)
    total_reward: float = 0.0
    final_test_mae: float = 0.0
    final_test_r2: float = 0.0
    contexts_covered: Set = field(default_factory=set)
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'env_name': self.env_name,
            'steps': self.steps,
            'total_reward': self.total_reward,
            'final_test_mae': self.final_test_mae,
            'final_test_r2': self.final_test_r2,
            'contexts_covered': list(self.contexts_covered),
            'metadata': self.metadata
        }


# ============================================================================
# REWARD FUNCTIONS
# ============================================================================

def reward_information_gain(
    old_pool_uncertainty: float,
    new_pool_uncertainty: float,
    prediction_error: float,
    alpha: float = 0.5
) -> float:
    """
    Reward based on information gain (uncertainty reduction).

    Args:
        old_pool_uncertainty: Mean uncertainty before query
        new_pool_uncertainty: Mean uncertainty after query
        prediction_error: |prediction - true_label|
        alpha: Weight for accuracy component (0 = pure info gain)

    Returns:
        Reward combining uncertainty reduction and accuracy
    """
    # Information gain = reduction in uncertainty
    info_gain = old_pool_uncertainty - new_pool_uncertainty

    # Accuracy bonus (negative error)
    accuracy = -prediction_error

    # Combine with weighting
    return (1 - alpha) * info_gain + alpha * accuracy


def reward_accuracy(prediction_error: float) -> float:
    """
    Simple accuracy-based reward.

    Args:
        prediction_error: |prediction - true_label|

    Returns:
        Negative prediction error
    """
    return -prediction_error


def reward_exploration_bonus(
    context: Tuple[int, int, int],
    context_counts: Dict[Tuple, int],
    base_reward: float,
    bonus_scale: float = 0.5
) -> float:
    """
    Add exploration bonus for underexplored contexts.

    Args:
        context: Context tuple (scaffold_cluster, mw_bin, logp_bin)
        context_counts: Dict mapping context to query count
        base_reward: Base reward before bonus
        bonus_scale: Scale factor for exploration bonus

    Returns:
        Reward with exploration bonus
    """
    count = context_counts.get(context, 0)

    # Inverse sqrt bonus: high for rare contexts, diminishing returns
    exploration_bonus = bonus_scale / np.sqrt(1 + count)

    return base_reward + exploration_bonus


def reward_hybrid(
    old_pool_uncertainty: float,
    new_pool_uncertainty: float,
    prediction_error: float,
    context: Tuple[int, int, int],
    context_counts: Dict[Tuple, int],
    weights: Dict[str, float] = None
) -> float:
    """
    Hybrid reward combining all components.

    Args:
        old_pool_uncertainty: Mean uncertainty before query
        new_pool_uncertainty: Mean uncertainty after query
        prediction_error: |prediction - true_label|
        context: Context tuple
        context_counts: Dict mapping context to query count
        weights: Dict with 'info_gain', 'accuracy', 'exploration' weights

    Returns:
        Weighted combination of all reward components
    """
    if weights is None:
        weights = {'info_gain': 0.4, 'accuracy': 0.4, 'exploration': 0.2}

    # Information gain
    info_gain = old_pool_uncertainty - new_pool_uncertainty

    # Accuracy (normalized to similar scale)
    accuracy = -prediction_error

    # Exploration bonus
    count = context_counts.get(context, 0)
    exploration = 1.0 / np.sqrt(1 + count)

    return (
        weights['info_gain'] * info_gain +
        weights['accuracy'] * accuracy +
        weights['exploration'] * exploration
    )


# ============================================================================
# MAIN ENVIRONMENT CLASS
# ============================================================================

class MolecularDesignEnv:
    """
    Agentic environment for molecular property exploration.

    State:
        - which molecules have been queried
        - current world model beliefs (predictions + uncertainties)
        - remaining budget

    Action:
        - select index of molecule to query from candidate pool

    Transition:
        - query oracle for true label
        - update world model beliefs
        - record episode step

    Episode:
        - sequence of (state, action, prediction, outcome, reward) tuples
        - this is what OC will consolidate
    """

    REWARD_TYPES = ['information_gain', 'accuracy', 'exploration', 'hybrid']

    def __init__(
        self,
        candidate_pool: pd.DataFrame = None,
        candidate_pool_path: str = None,
        test_set: pd.DataFrame = None,
        world_model: MolecularWorldModel = None,
        query_budget: int = 50,
        reward_type: str = 'hybrid',
        reward_weights: Dict[str, float] = None,
        random_state: int = 42,
        corruptor: Optional[DataCorruptor] = None
    ):
        """
        Initialize MolecularDesignEnv.

        Args:
            candidate_pool: DataFrame with 'smiles' and 'logS' columns
            candidate_pool_path: Path to load data (alternative to candidate_pool)
            test_set: Held-out test set for evaluation
            world_model: Pre-initialized MolecularWorldModel (or None to create new)
            query_budget: Maximum number of queries per episode
            reward_type: One of 'information_gain', 'accuracy', 'exploration', 'hybrid'
            reward_weights: Custom weights for hybrid reward
            random_state: Random seed
            corruptor: Optional DataCorruptor for stress testing
        """
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)
        self.corruptor = corruptor or DataCorruptor('none')

        # Load data
        if candidate_pool is not None:
            self.candidate_pool = candidate_pool.copy()
        elif candidate_pool_path is not None:
            data = load_esol_data(candidate_pool_path)
            self.candidate_pool = data['candidate_pool'].copy()
            if test_set is None:
                test_set = data['test_set']
        else:
            # Default: load ESOL data
            data = load_esol_data('data/esol_processed.pkl')
            self.candidate_pool = data['candidate_pool'].copy()
            if test_set is None:
                test_set = data['test_set']

        self.test_set = test_set
        self.n_candidates = len(self.candidate_pool)

        # Extract SMILES and labels
        self.all_smiles = self.candidate_pool['smiles'].tolist()
        self.oracle = self.candidate_pool['logS'].values  # True labels

        # World model
        self.base_world_model = world_model
        self.world_model = None  # Will be initialized in reset()

        # Budget and reward
        self.query_budget = query_budget
        self.reward_type = reward_type
        self.reward_weights = reward_weights or {
            'info_gain': 0.4,
            'accuracy': 0.4,
            'exploration': 0.2
        }

        # State variables (initialized in reset)
        self.queried_indices: Set[int] = set()
        self.remaining_budget: int = 0
        self.context_counts: Dict[Tuple, int] = {}
        self.episode: Episode = None
        self.current_step: int = 0

        # Track corrupted labels for each queried index
        # This allows OnlineModel to use the same corrupted data as FullStackModel
        self.corrupted_labels: Dict[int, float] = {}

        # Cache for predictions/uncertainties
        self._predictions_cache: Optional[np.ndarray] = None
        self._uncertainties_cache: Optional[np.ndarray] = None
        self._contexts_cache: Optional[List[Tuple]] = None

    def reset(self, seed_size: int = 10, seed_indices: List[int] = None) -> Dict:
        """
        Reset environment to fresh state.

        Args:
            seed_size: Number of random molecules to seed world model
            seed_indices: Specific indices to use as seed (overrides seed_size)

        Returns:
            Initial observation dict
        """
        # Reset random state
        self.rng = np.random.RandomState(self.random_state)

        # Reset corruptor
        self.corruptor.reset()

        # Reset state
        self.queried_indices = set()
        self.remaining_budget = self.query_budget
        self.context_counts = {}
        self.current_step = 0
        self.corrupted_labels = {}  # Reset corrupted labels tracking

        # Select seed molecules
        if seed_indices is not None:
            seed_idx = list(seed_indices)
        else:
            seed_idx = self.rng.choice(
                self.n_candidates,
                size=min(seed_size, self.n_candidates),
                replace=False
            ).tolist()

        # Mark seed as queried
        self.queried_indices.update(seed_idx)

        # Get seed data
        seed_smiles = [self.all_smiles[i] for i in seed_idx]
        seed_labels = [self.oracle[i] for i in seed_idx]

        # Initialize world model
        if self.base_world_model is not None:
            # Clone base model structure, retrain on seed
            self.world_model = MolecularWorldModel(
                n_scaffold_clusters=self.base_world_model.n_scaffold_clusters,
                n_mw_bins=self.base_world_model.n_mw_bins,
                n_logp_bins=self.base_world_model.n_logp_bins,
                n_estimators=self.base_world_model.n_estimators,
                random_state=self.random_state
            )
        else:
            self.world_model = MolecularWorldModel(random_state=self.random_state)

        # Fit on seed data
        self.world_model.fit(seed_smiles, seed_labels)

        # Invalidate cache
        self._invalidate_cache()

        # Update context counts for seed
        for smiles in seed_smiles:
            ctx = self.world_model.get_context(smiles)
            self.context_counts[ctx] = self.context_counts.get(ctx, 0) + 1

        # Initialize episode
        self.episode = Episode(
            metadata={
                'seed_size': len(seed_idx),
                'seed_indices': seed_idx,
                'query_budget': self.query_budget,
                'reward_type': self.reward_type,
                'reward_weights': self.reward_weights,
                'n_candidates': self.n_candidates
            }
        )

        return self.get_state()

    def _invalidate_cache(self):
        """Invalidate prediction cache (call after world model update)."""
        self._predictions_cache = None
        self._uncertainties_cache = None

    def _update_cache(self):
        """Update prediction cache if needed."""
        if self._predictions_cache is None:
            preds, uncerts = self.world_model.predict(
                self.all_smiles,
                return_uncertainty=True
            )
            self._predictions_cache = preds
            self._uncertainties_cache = uncerts

        if self._contexts_cache is None:
            self._contexts_cache = self.world_model.get_contexts(self.all_smiles)

    def get_state(self) -> Dict[str, Any]:
        """
        Get current environment state.

        Returns:
            Dict with:
            - queried_indices: list of queried molecule indices
            - unqueried_indices: list of unqueried molecule indices
            - predictions: world model predictions for ALL molecules
            - uncertainties: world model uncertainties for ALL molecules
            - remaining_budget: int
            - contexts: context labels for all molecules
            - context_counts: dict of context -> query count
        """
        self._update_cache()

        unqueried = [i for i in range(self.n_candidates) if i not in self.queried_indices]

        # Apply candidate filtering for shift/imbalance corruption modes
        # This restricts which molecules can be selected, simulating distribution shift
        if self.corruptor.mode in ['shift', 'imbalance']:
            contexts_for_unqueried = [self._contexts_cache[i] for i in unqueried]
            unqueried = self.corruptor.filter_candidates(unqueried, contexts_for_unqueried)

        return {
            'queried_indices': list(self.queried_indices),
            'unqueried_indices': unqueried,
            'predictions': self._predictions_cache.copy(),
            'uncertainties': self._uncertainties_cache.copy(),
            'remaining_budget': self.remaining_budget,
            'contexts': self._contexts_cache.copy(),
            'context_counts': self.context_counts.copy(),
            'step': self.current_step
        }

    def get_unqueried_mask(self) -> np.ndarray:
        """Get boolean mask for unqueried molecules."""
        mask = np.ones(self.n_candidates, dtype=bool)
        for idx in self.queried_indices:
            mask[idx] = False
        return mask

    def get_pool_uncertainty(self, mask: np.ndarray = None) -> float:
        """Get mean uncertainty for unqueried molecules."""
        self._update_cache()

        if mask is None:
            mask = self.get_unqueried_mask()

        unqueried_uncerts = self._uncertainties_cache[mask]
        valid_uncerts = unqueried_uncerts[~np.isnan(unqueried_uncerts)]

        if len(valid_uncerts) == 0:
            return 0.0
        return float(np.mean(valid_uncerts))

    def step(self, action_idx: int) -> Tuple[Dict, float, bool, Dict]:
        """
        Query molecule at action_idx.

        Args:
            action_idx: Index in candidate pool to query

        Returns:
            observation: dict with query result and new state
            reward: float
            done: bool (budget exhausted or all queried)
            info: dict with diagnostics
        """
        # Validate action
        if action_idx in self.queried_indices:
            raise ValueError(f"Molecule {action_idx} already queried")
        if action_idx < 0 or action_idx >= self.n_candidates:
            raise ValueError(f"Invalid action index: {action_idx}")
        if self.remaining_budget <= 0:
            raise ValueError("No remaining budget")

        smiles = self.all_smiles[action_idx]

        # 1. Get world model prediction BEFORE seeing true label
        self._update_cache()
        pred = self._predictions_cache[action_idx]
        unc = self._uncertainties_cache[action_idx]
        context = self._contexts_cache[action_idx]

        # Get pool uncertainty before update
        old_pool_uncertainty = self.get_pool_uncertainty()

        # 2. Query oracle (true label)
        true_label = self.oracle[action_idx]

        # 2.5. Apply corruption (for stress testing)
        # World model sees corrupted label, but we evaluate on TRUE labels
        corrupted_label = self.corruptor.corrupt_label(
            true_label,
            step=self.current_step,
            context=context
        )

        # 3. Mark as queried and store corrupted label
        self.queried_indices.add(action_idx)
        self.remaining_budget -= 1
        self.corrupted_labels[action_idx] = corrupted_label  # Track for OnlineModel

        # 4. Update world model with CORRUPTED data point
        # This simulates noisy/biased labels in real-world scenarios
        self.world_model.update([smiles], [corrupted_label])
        self._invalidate_cache()

        # Get new pool uncertainty
        new_pool_uncertainty = self.get_pool_uncertainty()

        # 5. Compute reward
        prediction_error = abs(pred - true_label)
        reward = self._compute_reward(
            old_pool_uncertainty,
            new_pool_uncertainty,
            prediction_error,
            context
        )

        # Update context counts
        self.context_counts[context] = self.context_counts.get(context, 0) + 1

        # 6. Record step
        self.current_step += 1
        cumulative_reward = self.episode.total_reward + reward
        self.episode.total_reward = cumulative_reward

        step_record = {
            'step': self.current_step,
            'action': action_idx,
            'smiles': smiles,
            'context': context,
            'prediction': float(pred),
            'uncertainty': float(unc),
            'true_label': float(true_label),
            'corrupted_label': float(corrupted_label),
            'was_corrupted': abs(true_label - corrupted_label) > 1e-6,
            'error': float(prediction_error),
            'reward': float(reward),
            'cumulative_reward': float(cumulative_reward),
            'remaining_budget': self.remaining_budget,
            'pool_mean_uncertainty': float(new_pool_uncertainty)
        }
        self.episode.steps.append(step_record)
        self.episode.contexts_covered.add(context)

        # 7. Check if done
        done = (self.remaining_budget <= 0 or
                len(self.queried_indices) >= self.n_candidates)

        # 8. Build observation
        observation = {
            'query_result': {
                'smiles': smiles,
                'prediction': pred,
                'true_label': true_label,
                'error': prediction_error,
                'uncertainty': unc,
                'context': context
            },
            'state': self.get_state()
        }

        # 9. Info dict
        info = {
            'step': self.current_step,
            'old_pool_uncertainty': old_pool_uncertainty,
            'new_pool_uncertainty': new_pool_uncertainty,
            'uncertainty_reduction': old_pool_uncertainty - new_pool_uncertainty,
            'n_queried': len(self.queried_indices),
            'n_contexts_covered': len(self.episode.contexts_covered)
        }

        return observation, reward, done, info

    def _compute_reward(
        self,
        old_pool_uncertainty: float,
        new_pool_uncertainty: float,
        prediction_error: float,
        context: Tuple[int, int, int]
    ) -> float:
        """Compute reward based on reward_type."""
        if self.reward_type == 'information_gain':
            return reward_information_gain(
                old_pool_uncertainty,
                new_pool_uncertainty,
                prediction_error,
                alpha=0.3
            )
        elif self.reward_type == 'accuracy':
            return reward_accuracy(prediction_error)
        elif self.reward_type == 'exploration':
            base = reward_accuracy(prediction_error)
            return reward_exploration_bonus(
                context,
                self.context_counts,
                base,
                bonus_scale=0.5
            )
        elif self.reward_type == 'hybrid':
            return reward_hybrid(
                old_pool_uncertainty,
                new_pool_uncertainty,
                prediction_error,
                context,
                self.context_counts,
                self.reward_weights
            )
        else:
            raise ValueError(f"Unknown reward type: {self.reward_type}")

    def get_episode(self) -> Dict:
        """
        Get episode trajectory for OC consumption.

        Returns:
            Episode dict with steps and metadata
        """
        # Evaluate on test set if available
        if self.test_set is not None:
            test_smiles = self.test_set['smiles'].tolist()
            test_labels = self.test_set['logS'].values
            metrics = self.evaluate_on_test(test_smiles, test_labels)
            self.episode.final_test_mae = metrics['mae']
            self.episode.final_test_r2 = metrics['r2']

        return self.episode.to_dict()

    def evaluate_on_test(
        self,
        test_smiles: List[str],
        test_labels: Union[List[float], np.ndarray]
    ) -> Dict:
        """
        Evaluate current world model on held-out test set.

        Args:
            test_smiles: Test SMILES strings
            test_labels: True property values

        Returns:
            Dict with MAE, RMSE, R², calibration metrics
        """
        return self.world_model.get_calibration_metrics(test_smiles, test_labels)

    def get_valid_actions(self) -> List[int]:
        """Get list of valid action indices (unqueried molecules)."""
        return [i for i in range(self.n_candidates) if i not in self.queried_indices]

    def render(self, mode: str = 'text') -> str:
        """Render current state as text."""
        self._update_cache()

        lines = [
            f"=== MolecularDesignEnv ===",
            f"Step: {self.current_step}",
            f"Remaining budget: {self.remaining_budget}",
            f"Queried: {len(self.queried_indices)}/{self.n_candidates}",
            f"Contexts covered: {len(self.episode.contexts_covered) if self.episode else 0}",
            f"Total reward: {self.episode.total_reward:.4f}" if self.episode else "",
            f"Pool mean uncertainty: {self.get_pool_uncertainty():.4f}",
        ]

        if self.episode and len(self.episode.steps) > 0:
            last_step = self.episode.steps[-1]
            lines.extend([
                f"\nLast query:",
                f"  SMILES: {last_step['smiles'][:50]}...",
                f"  Predicted: {last_step['prediction']:.3f}",
                f"  True: {last_step['true_label']:.3f}",
                f"  Error: {last_step['error']:.3f}",
                f"  Reward: {last_step['reward']:.4f}"
            ])

        return '\n'.join(lines)


# ============================================================================
# POLICIES FOR TESTING
# ============================================================================

class Policy:
    """Base class for action selection policies."""

    def select_action(self, state: Dict) -> int:
        raise NotImplementedError


class RandomPolicy(Policy):
    """Randomly select from unqueried molecules."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)

    def select_action(self, state: Dict) -> int:
        unqueried = state['unqueried_indices']
        return self.rng.choice(unqueried)


class UncertaintySamplingPolicy(Policy):
    """Select molecule with highest uncertainty."""

    def select_action(self, state: Dict) -> int:
        unqueried = state['unqueried_indices']
        uncertainties = state['uncertainties']

        # Get uncertainties for unqueried only
        unqueried_uncerts = [(i, uncertainties[i]) for i in unqueried]

        # Filter out NaN
        valid = [(i, u) for i, u in unqueried_uncerts if not np.isnan(u)]

        if not valid:
            return unqueried[0]

        # Select highest uncertainty
        return max(valid, key=lambda x: x[1])[0]


class GreedyAccuracyPolicy(Policy):
    """Select molecule where we're most confident (lowest uncertainty)."""

    def select_action(self, state: Dict) -> int:
        unqueried = state['unqueried_indices']
        uncertainties = state['uncertainties']

        unqueried_uncerts = [(i, uncertainties[i]) for i in unqueried]
        valid = [(i, u) for i, u in unqueried_uncerts if not np.isnan(u)]

        if not valid:
            return unqueried[0]

        return min(valid, key=lambda x: x[1])[0]


class ContextDiversityPolicy(Policy):
    """Select molecule from least-explored context."""

    def select_action(self, state: Dict) -> int:
        unqueried = state['unqueried_indices']
        contexts = state['contexts']
        context_counts = state['context_counts']

        # Group unqueried by context
        context_to_indices = {}
        for i in unqueried:
            ctx = contexts[i]
            if ctx not in context_to_indices:
                context_to_indices[ctx] = []
            context_to_indices[ctx].append(i)

        # Find least explored context
        min_count = float('inf')
        best_context = None

        for ctx, indices in context_to_indices.items():
            count = context_counts.get(ctx, 0)
            if count < min_count:
                min_count = count
                best_context = ctx

        if best_context is None:
            return unqueried[0]

        # Return first molecule from that context
        return context_to_indices[best_context][0]


class HybridExplorationPolicy(Policy):
    """
    Balance uncertainty sampling with context diversity.

    Score = uncertainty * (1 + exploration_bonus)
    """

    def __init__(self, exploration_weight: float = 0.5):
        self.exploration_weight = exploration_weight

    def select_action(self, state: Dict) -> int:
        unqueried = state['unqueried_indices']
        uncertainties = state['uncertainties']
        contexts = state['contexts']
        context_counts = state['context_counts']

        best_idx = None
        best_score = -float('inf')

        for i in unqueried:
            unc = uncertainties[i]
            if np.isnan(unc):
                continue

            ctx = contexts[i]
            count = context_counts.get(ctx, 0)

            # Exploration bonus
            exploration = 1.0 / np.sqrt(1 + count)

            # Combined score
            score = unc + self.exploration_weight * exploration

            if score > best_score:
                best_score = score
                best_idx = i

        return best_idx if best_idx is not None else unqueried[0]


def run_episode(
    env: MolecularDesignEnv,
    policy: Policy,
    seed_size: int = 10,
    verbose: bool = False
) -> Dict:
    """
    Run a complete episode with given policy.

    Args:
        env: MolecularDesignEnv instance
        policy: Action selection policy
        seed_size: Number of seed molecules
        verbose: Print progress

    Returns:
        Episode dict
    """
    state = env.reset(seed_size=seed_size)

    if verbose:
        print(f"Starting episode with {seed_size} seed molecules")
        print(f"Budget: {env.query_budget}, Candidates: {env.n_candidates}")

    done = False
    while not done:
        action = policy.select_action(state)
        obs, reward, done, info = env.step(action)
        state = obs['state']

        if verbose and env.current_step % 10 == 0:
            print(f"  Step {env.current_step}: "
                  f"error={obs['query_result']['error']:.3f}, "
                  f"reward={reward:.4f}, "
                  f"pool_unc={info['new_pool_uncertainty']:.4f}")

    episode = env.get_episode()

    if verbose:
        print(f"\nEpisode complete:")
        print(f"  Total reward: {episode['total_reward']:.4f}")
        print(f"  Final test MAE: {episode['final_test_mae']:.4f}")
        print(f"  Contexts covered: {len(episode['contexts_covered'])}")

    return episode
