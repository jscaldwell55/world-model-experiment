"""
Experiment Metrics and Logging for OC+FTB

Tracks comprehensive metrics for offline consolidation and fine-tuning bridge experiments.

Metrics tracked:
- Episode counts (real vs synthetic)
- Context distributions
- Accuracy before/after
- Cross-validation error
- Uncertainty calibration
- Computational costs
"""

import json
import time
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class ExperimentMetrics:
    """Comprehensive metrics for a single OC+FTB experiment run"""

    # ===== Configuration =====
    domain: str
    phase: str  # 'phase0', 'phase1', etc.
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    config: Optional[Dict] = None

    # ===== Episode Counts =====
    episodes_before: int = 0
    episodes_after: int = 0
    synthetics_generated: int = 0
    synthetics_saved: int = 0
    synthetics_filtered_out: int = 0

    # ===== Context Distribution =====
    context_distribution_before: Dict[str, int] = field(default_factory=dict)
    context_distribution_after: Dict[str, int] = field(default_factory=dict)

    # ===== Model Quality =====
    cv_error: float = 1.0
    gate_status: str = "PENDING"  # 'PASS', 'WARNING', 'FAIL'
    gate_reason: str = ""

    # ===== Performance Metrics (if measured) =====
    accuracy_before: Optional[float] = None
    accuracy_after: Optional[float] = None
    accuracy_delta: Optional[float] = None

    # ===== Uncertainty Calibration (Phase 1) =====
    uncertainty_correlation: Optional[float] = None
    uncertainty_p_value: Optional[float] = None
    brier_score: Optional[float] = None
    mean_uncertainty: Optional[float] = None
    mean_error: Optional[float] = None

    # ===== Fidelity Scores =====
    mean_fidelity: Optional[float] = None
    min_fidelity: Optional[float] = None
    max_fidelity: Optional[float] = None
    fidelity_std: Optional[float] = None

    # ===== Computational Cost =====
    oc_runtime_seconds: Optional[float] = None
    ftb_runtime_seconds: Optional[float] = None
    total_runtime_seconds: Optional[float] = None

    # ===== Additional Metadata =====
    notes: str = ""
    errors: List[str] = field(default_factory=list)

    def compute_derived_metrics(self):
        """Compute derived metrics from base measurements"""

        # Accuracy delta
        if self.accuracy_before is not None and self.accuracy_after is not None:
            self.accuracy_delta = self.accuracy_after - self.accuracy_before

        # Filter rate
        if self.synthetics_generated > 0:
            self.synthetics_filtered_out = self.synthetics_generated - self.synthetics_saved

        # Total runtime
        if self.oc_runtime_seconds is not None and self.ftb_runtime_seconds is not None:
            self.total_runtime_seconds = self.oc_runtime_seconds + self.ftb_runtime_seconds

    def save(self, output_dir: str = "results/metrics"):
        """
        Save metrics to JSON file.

        Args:
            output_dir: Directory to save metrics
        """
        # Compute derived metrics before saving
        self.compute_derived_metrics()

        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Generate filename
        timestamp_str = datetime.fromisoformat(self.timestamp).strftime("%Y%m%d_%H%M%S")
        filename = f"{self.domain}_{self.phase}_{timestamp_str}.json"
        filepath = Path(output_dir) / filename

        # Save to JSON
        with open(filepath, 'w') as f:
            json.dump(asdict(self), f, indent=2)

        logger.info(f"✓ Saved experiment metrics to {filepath}")

        return filepath

    def print_summary(self):
        """Pretty-print metrics summary to console"""

        # Compute derived metrics
        self.compute_derived_metrics()

        print(f"\n{'='*70}")
        print(f"Experiment Metrics: {self.domain.upper()} ({self.phase})")
        print(f"Timestamp: {self.timestamp}")
        print(f"{'='*70}")

        # Episodes
        print(f"\n📊 EPISODES:")
        print(f"  Before:             {self.episodes_before}")
        print(f"  After:              {self.episodes_after}")
        print(f"  Synthetics created: {self.synthetics_generated}")
        print(f"  Synthetics saved:   {self.synthetics_saved}")
        if self.synthetics_filtered_out > 0:
            print(f"  Filtered out:       {self.synthetics_filtered_out}")

        # Model Quality
        print(f"\n🎯 MODEL QUALITY:")
        print(f"  CV Error:           {self.cv_error:.1%}")
        print(f"  Gate Status:        {self.gate_status}")
        if self.gate_reason:
            print(f"  Reason:             {self.gate_reason}")

        # Accuracy (if available)
        if self.accuracy_before is not None and self.accuracy_after is not None:
            print(f"\n📈 ACCURACY:")
            print(f"  Before:             {self.accuracy_before:.1%}")
            print(f"  After:              {self.accuracy_after:.1%}")
            delta_sign = "+" if self.accuracy_delta >= 0 else ""
            print(f"  Delta:              {delta_sign}{self.accuracy_delta:.1%}")

        # Fidelity (if available)
        if self.mean_fidelity is not None:
            print(f"\n🔬 SYNTHETIC FIDELITY:")
            print(f"  Mean:               {self.mean_fidelity:.3f}")
            if self.min_fidelity is not None and self.max_fidelity is not None:
                print(f"  Range:              [{self.min_fidelity:.3f}, {self.max_fidelity:.3f}]")
            if self.fidelity_std is not None:
                print(f"  Std Dev:            {self.fidelity_std:.3f}")

        # Uncertainty Calibration (Phase 1)
        if self.uncertainty_correlation is not None:
            print(f"\n🎲 UNCERTAINTY CALIBRATION:")
            print(f"  Unc-Error Correlation: {self.uncertainty_correlation:.3f}", end="")
            if self.uncertainty_p_value is not None:
                sig = " ***" if self.uncertainty_p_value < 0.001 else " **" if self.uncertainty_p_value < 0.01 else " *" if self.uncertainty_p_value < 0.05 else ""
                print(f" (p={self.uncertainty_p_value:.4f}{sig})")
            else:
                print()

            if self.brier_score is not None:
                print(f"  Brier Score:           {self.brier_score:.3f}")
            if self.mean_uncertainty is not None:
                print(f"  Mean Uncertainty:      {self.mean_uncertainty:.2f}")
            if self.mean_error is not None:
                print(f"  Mean Error:            {self.mean_error:.2f}")

        # Context Distribution
        if self.context_distribution_after:
            print(f"\n🌍 CONTEXT DISTRIBUTION (After):")
            total_after = sum(self.context_distribution_after.values())
            for context, count in sorted(self.context_distribution_after.items()):
                pct = 100 * count / total_after if total_after > 0 else 0
                print(f"  {str(context):20s}: {count:4d} ({pct:5.1f}%)")

        # Runtime
        if self.oc_runtime_seconds is not None or self.ftb_runtime_seconds is not None:
            print(f"\n⏱️  RUNTIME:")
            if self.oc_runtime_seconds is not None:
                print(f"  OC:                 {self.oc_runtime_seconds:7.1f}s")
            if self.ftb_runtime_seconds is not None:
                print(f"  FTB:                {self.ftb_runtime_seconds:7.1f}s")
            if self.total_runtime_seconds is not None:
                print(f"  Total:              {self.total_runtime_seconds:7.1f}s")

        # Errors
        if self.errors:
            print(f"\n⚠️  ERRORS ({len(self.errors)}):")
            for error in self.errors[:5]:  # Show first 5
                print(f"  • {error}")
            if len(self.errors) > 5:
                print(f"  ... and {len(self.errors) - 5} more")

        # Notes
        if self.notes:
            print(f"\n📝 NOTES:")
            for line in self.notes.split('\n'):
                print(f"  {line}")

        print(f"{'='*70}\n")


def compute_context_distribution(playbook, context_spec) -> Dict[str, int]:
    """
    Count episodes per context in a playbook.

    Args:
        playbook: ACEPlaybook instance
        context_spec: ContextSpec for the domain

    Returns:
        Dictionary mapping context keys to counts
    """
    from collections import Counter

    contexts = []

    # Get observations from playbook
    observations = playbook.playbook.get('observations', [])

    for obs in observations:
        try:
            context_key = context_spec.extract_context(obs)
            contexts.append(str(context_key))  # Convert to string for JSON serialization
        except Exception:
            contexts.append("UNKNOWN")

    distribution = dict(Counter(contexts))

    return distribution


def compute_fidelity_stats(synthetic_episodes: List[dict]) -> Dict[str, float]:
    """
    Compute fidelity statistics for synthetic episodes.

    Args:
        synthetic_episodes: List of synthetic episode dicts with 'fidelity_score'

    Returns:
        Dictionary with mean, min, max, std of fidelity scores
    """
    import numpy as np

    fidelity_scores = [
        ep.get('fidelity_score', 0.0)
        for ep in synthetic_episodes
    ]

    if not fidelity_scores:
        return {
            'mean_fidelity': None,
            'min_fidelity': None,
            'max_fidelity': None,
            'fidelity_std': None
        }

    return {
        'mean_fidelity': float(np.mean(fidelity_scores)),
        'min_fidelity': float(np.min(fidelity_scores)),
        'max_fidelity': float(np.max(fidelity_scores)),
        'fidelity_std': float(np.std(fidelity_scores))
    }


def load_metrics(filepath: str) -> ExperimentMetrics:
    """
    Load metrics from JSON file.

    Args:
        filepath: Path to metrics JSON file

    Returns:
        ExperimentMetrics instance
    """
    with open(filepath, 'r') as f:
        data = json.load(f)

    return ExperimentMetrics(**data)


def compare_experiments(metrics_list: List[ExperimentMetrics]):
    """
    Print comparison table of multiple experiments.

    Args:
        metrics_list: List of ExperimentMetrics to compare
    """
    if not metrics_list:
        print("No metrics to compare")
        return

    print(f"\n{'='*70}")
    print("EXPERIMENT COMPARISON")
    print(f"{'='*70}\n")

    # Table header
    print(f"{'Domain':15s} {'Phase':10s} {'CV Error':10s} {'Gate':8s} {'Synthetics':11s} {'Accuracy Δ':12s}")
    print("-" * 70)

    # Table rows
    for m in sorted(metrics_list, key=lambda x: (x.domain, x.phase)):
        cv_err = f"{m.cv_error:.1%}" if m.cv_error < 1.0 else "N/A"
        gate = m.gate_status[:8]
        synth = f"{m.synthetics_saved}/{m.synthetics_generated}"
        acc_delta = f"{m.accuracy_delta:+.1%}" if m.accuracy_delta is not None else "N/A"

        print(f"{m.domain:15s} {m.phase:10s} {cv_err:10s} {gate:8s} {synth:11s} {acc_delta:12s}")

    print()
