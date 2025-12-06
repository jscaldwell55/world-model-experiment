"""
Molecular Consolidation Pipeline with Fine-Tuning Bridge (FTB).

Architecture V2.0: Replaces the old Offline Consolidation (OC) gate approach
with an FTB that accepts all data but weights it, using Replay Repair to
prevent catastrophic forgetting.

Key Mechanism:
- If FTB detects forgetting (probe MAE increases), it triggers a Replay Repair
  loop that mixes old memories (probe set) with new data to restore performance.

Metric Definitions:
- improvement_ratio: pre_update_mae / post_update_mae
  - > 1.0: Model improved on probe set (good)
  - = 1.0: No change
  - < 1.0: Model degraded (potential forgetting)
  - Threshold check: improvement_ratio >= (1.0 - retention_threshold)
    e.g., with threshold=0.25, we allow up to 25% degradation (ratio >= 0.75)
"""

import numpy as np
from typing import List, Dict, Optional, Tuple


class SimplifiedFTB:
    """
    Fine-Tuning Bridge (FTB) with Replay Repair.

    Features:
    1. Batched Retraining: Updates model on accumulated data.
    2. Retention Check: Validates performance on a 'Probe Set' (past knowledge).
    3. Replay Repair: If retention fails, augments the training batch with
       data from the Probe Set to force the model to 'remember'.
    4. Soft Weighting: Supports sample weights for reliability.

    Metrics:
    - improvement_ratio: pre_mae / post_mae (>1 = improved, <1 = degraded)
    - retention_passed: True if improvement_ratio >= (1 - threshold)
    """

    def __init__(self,
                 world_model,
                 probe_smiles: List[str],
                 probe_labels: List[float],
                 retention_threshold: float = 0.25,
                 repair_attempts: int = 3,
                 replay_ratio: float = 0.5,
                 random_state: int = 42):
        """
        Args:
            world_model: The model to train. Must support .fit(X, y, sample_weight=...).
            probe_smiles: Representative data from past episodes (to check forgetting).
            probe_labels: Labels for probe data.
            retention_threshold: Allowed fractional degradation (e.g., 0.25 = 25% MAE increase allowed).
            repair_attempts: Max times to try repairing the model if retention fails.
            replay_ratio: Ratio of probe data to add during repair (relative to batch size).
        """
        self.world_model = world_model
        self.probe_smiles = np.array(probe_smiles)
        self.probe_labels = np.array(probe_labels)
        self.retention_threshold = retention_threshold
        self.repair_attempts = repair_attempts
        self.replay_ratio = replay_ratio
        self.rng = np.random.RandomState(random_state)

        # Initialize baseline
        self.baseline_mae = self.compute_probe_mae()

    def compute_probe_mae(self) -> float:
        """Evaluate current model state on the probe set."""
        preds, _ = self.world_model.predict(self.probe_smiles)
        return np.mean(np.abs(preds - self.probe_labels))

    def _get_replay_batch(self, n_needed: int) -> Tuple[List[str], List[float]]:
        """Sample 'n_needed' items from the probe set to act as memory replay."""
        if len(self.probe_smiles) == 0:
            return [], []
        indices = self.rng.choice(len(self.probe_smiles), size=n_needed, replace=True)
        return self.probe_smiles[indices].tolist(), self.probe_labels[indices].tolist()

    def update(self,
               smiles: List[str],
               labels: List[float],
               weights: Optional[List[float]] = None) -> Dict:
        """
        Retrain world model with retention guarantees.

        Returns:
            Dict with:
            - pre_update_mae: MAE on probe set before update
            - post_update_mae: MAE on probe set after update
            - improvement_ratio: pre/post MAE (>1 = improved, <1 = degraded)
            - retention_passed: True if no catastrophic forgetting detected
            - was_repaired: True if replay repair was triggered
            - repair_attempts: Number of repair attempts made
            - repair_details: Log of each repair attempt
            - n_samples: Number of training samples
        """
        # 1. Snapshot State
        pre_update_mae = self.compute_probe_mae()

        # Default weights = 1.0 if not provided
        if weights is None:
            weights = [1.0] * len(smiles)

        # 2. Initial Attempt: Standard Fit
        self.world_model.fit(smiles, labels, sample_weight=weights)

        # 3. Check Retention
        post_update_mae = self.compute_probe_mae()
        initial_post_mae = post_update_mae  # Store for diagnostics (before any repair)

        # Compute improvement_ratio: pre/post
        # > 1.0 means improvement (post_mae decreased)
        # < 1.0 means degradation (post_mae increased)
        if post_update_mae > 0:
            improvement_ratio = pre_update_mae / post_update_mae
        else:
            improvement_ratio = 1.0  # Perfect prediction, no degradation

        initial_ratio = improvement_ratio  # Store for diagnostics (before any repair)

        # Check if within acceptable threshold
        # threshold=0.25 means we allow ratio as low as 0.75 (25% degradation)
        retention_passed = improvement_ratio >= (1.0 - self.retention_threshold)

        repair_log = []

        # 4. Repair Loop (if retention failed)
        if not retention_passed:
            # We are in a "Forgetting" state. Initiate Replay.

            for attempt in range(self.repair_attempts):
                # Calculate how much replay data to mix in
                n_replay = int(len(smiles) * self.replay_ratio * (attempt + 1))
                if n_replay == 0:
                    n_replay = 1

                replay_smiles, replay_labels = self._get_replay_batch(n_replay)

                # Create mixed batch
                mixed_smiles = smiles + replay_smiles
                mixed_labels = labels + replay_labels

                # Give replay data slightly higher weight to force memory
                # Weights: Original data gets passed weights, Replay data gets 1.5
                mixed_weights = weights + [1.5] * len(replay_labels)

                # Retrain
                self.world_model.fit(mixed_smiles, mixed_labels, sample_weight=mixed_weights)

                # Check again
                new_mae = self.compute_probe_mae()
                if new_mae > 0:
                    new_ratio = pre_update_mae / new_mae
                else:
                    new_ratio = 1.0

                passed = new_ratio >= (1.0 - self.retention_threshold)

                repair_log.append({
                    'attempt': attempt + 1,
                    'n_replay': n_replay,
                    'mae': new_mae,
                    'improvement_ratio': new_ratio,
                    'passed': passed
                })

                if passed:
                    post_update_mae = new_mae
                    improvement_ratio = new_ratio
                    retention_passed = True
                    break

        return {
            'pre_update_mae': pre_update_mae,
            'post_update_mae': post_update_mae,
            'initial_post_mae': initial_post_mae,  # MAE right after initial fit (before repair)
            'initial_ratio': initial_ratio,  # Ratio that triggered repair (if any)
            'improvement_ratio': improvement_ratio,  # Final ratio after all repairs
            'retention_passed': retention_passed,
            'was_repaired': len(repair_log) > 0,
            'repair_attempts': len(repair_log),
            'repair_details': repair_log,
            'n_samples': len(smiles)
        }
