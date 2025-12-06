"""
MolecularOCAdapter: Adapts molecular episodes to Offline Consolidation format.

This module bridges MolecularDesignEnv episodes to the existing OC system,
enabling bias detection, coverage analysis, and quality gating for molecular
property prediction trajectories.

Key differences from toy domains:
- Contexts are (scaffold_cluster, mw_bin, logp_bin) tuples
- Beliefs are property predictions with uncertainties
- Episodes are query sequences, not single observations
"""

import numpy as np
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Hashable, Any
from collections import defaultdict

from utils.context_spec import ContextSpec


# ============================================================================
# MOLECULAR CONTEXT SPECIFICATION
# ============================================================================

def _molecular_context_extractor(obs: dict) -> Hashable:
    """
    Extract molecular context from observation/step.

    Context = (scaffold_cluster, mw_bin, logp_bin)
    """
    if 'context' in obs:
        ctx = obs['context']
        if isinstance(ctx, tuple):
            return ctx
        elif isinstance(ctx, dict):
            return (
                ctx.get('scaffold_cluster', 0),
                ctx.get('mw_bin', 0),
                ctx.get('logp_bin', 0)
            )
        elif isinstance(ctx, list):
            return tuple(ctx)
        else:
            return ctx
    else:
        # Try to extract from step data
        return (
            obs.get('scaffold_cluster', 0),
            obs.get('mw_bin', 0),
            obs.get('logp_bin', 0)
        )


MOLECULAR_CONTEXT = ContextSpec(
    name="molecular_design",
    key_fn=_molecular_context_extractor
)


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class MolecularOCInput:
    """Input format for molecular OC analysis."""
    episodes: List[Dict]
    domain: str = 'molecular_design'
    context_spec: ContextSpec = field(default_factory=lambda: MOLECULAR_CONTEXT)


@dataclass
class MolecularOCOutput:
    """Output from molecular OC analysis."""
    # Quality gate
    passes_quality_gate: bool = False
    gate_reason: str = ""
    gate_status: str = "PENDING"  # 'PASS' | 'WARNING' | 'FAIL'

    # Diagnostics
    diagnostics: Dict = field(default_factory=dict)

    # Curated data for FTB
    curated_steps: List[Dict] = field(default_factory=list)
    curated_smiles: List[str] = field(default_factory=list)
    curated_labels: List[float] = field(default_factory=list)
    curated_contexts: List[Tuple] = field(default_factory=list)

    # Coverage analysis
    context_coverage: Dict = field(default_factory=dict)
    bias_report: Dict = field(default_factory=dict)

    # Calibration analysis
    calibration_metrics: Dict = field(default_factory=dict)


@dataclass
class MolecularFTBInput:
    """Input format for molecular FTB."""
    smiles: List[str]
    labels: List[float]
    contexts: List[Tuple]
    weights: List[float]  # Quality weights for each sample
    quality_metrics: Dict


# ============================================================================
# MAIN ADAPTER CLASS
# ============================================================================

class MolecularOCAdapter:
    """
    Adapts molecular episodes to OC's expected format.

    Converts MolecularDesignEnv episodes (query trajectories) into
    the format expected by the existing OC system.
    """

    def __init__(
        self,
        min_steps_per_episode: int = 5,  # Lowered from 10 - we validate episode quality elsewhere
        min_episodes: int = 1,
        max_error_threshold: float = 2.0,  # Filter out steps with huge errors
        calibration_threshold: float = 0.0  # Min correlation for uncertainty
    ):
        """
        Initialize adapter with quality thresholds.

        Args:
            min_steps_per_episode: Minimum steps to consider an episode valid
            min_episodes: Minimum episodes needed for consolidation
            max_error_threshold: Maximum |error| to include a step
            calibration_threshold: Minimum uncertainty-error correlation
        """
        self.min_steps_per_episode = min_steps_per_episode
        self.min_episodes = min_episodes
        self.max_error_threshold = max_error_threshold
        self.calibration_threshold = calibration_threshold

    def episodes_to_oc_format(self, episodes: List[Dict]) -> MolecularOCInput:
        """
        Convert MolecularDesignEnv episodes to OC input format.

        Args:
            episodes: List of episode dicts from MolecularDesignEnv.get_episode()

        Returns:
            MolecularOCInput ready for OC analysis
        """
        # Validate episodes
        valid_episodes = []
        for ep in episodes:
            if self._validate_episode(ep):
                valid_episodes.append(ep)
            else:
                warnings.warn(f"Invalid episode skipped: {ep.get('metadata', {})}")

        return MolecularOCInput(
            episodes=valid_episodes,
            domain='molecular_design',
            context_spec=MOLECULAR_CONTEXT
        )

    def _validate_episode(self, episode: Dict) -> bool:
        """Check if episode has required fields."""
        required = ['steps', 'total_reward', 'metadata']
        if not all(k in episode for k in required):
            return False

        steps = episode.get('steps', [])
        if len(steps) < self.min_steps_per_episode:
            return False

        # Check step structure
        step_required = ['smiles', 'context', 'prediction', 'true_label']
        for step in steps[:3]:  # Check first few steps
            if not all(k in step for k in step_required):
                return False

        return True

    def analyze(self, oc_input: MolecularOCInput) -> MolecularOCOutput:
        """
        Run OC analysis on molecular episodes.

        Args:
            oc_input: MolecularOCInput from episodes_to_oc_format()

        Returns:
            MolecularOCOutput with diagnostics and curated data
        """
        output = MolecularOCOutput()

        # Check minimum data requirements
        if len(oc_input.episodes) < self.min_episodes:
            output.gate_status = 'FAIL'
            output.gate_reason = f"Insufficient episodes: {len(oc_input.episodes)} < {self.min_episodes}"
            output.passes_quality_gate = False
            return output

        # Extract all steps from episodes
        all_steps = []
        for ep in oc_input.episodes:
            all_steps.extend(ep.get('steps', []))

        if not all_steps:
            output.gate_status = 'FAIL'
            output.gate_reason = "No steps found in episodes"
            output.passes_quality_gate = False
            return output

        # 1. Analyze coverage
        output.context_coverage = self._analyze_coverage(all_steps)

        # 2. Detect biases
        output.bias_report = self._detect_biases(all_steps, oc_input.episodes)

        # 3. Analyze calibration
        output.calibration_metrics = self._analyze_calibration(all_steps)

        # 4. Curate steps (filter by quality)
        curated = self._curate_steps(all_steps)
        output.curated_steps = curated['steps']
        output.curated_smiles = curated['smiles']
        output.curated_labels = curated['labels']
        output.curated_contexts = curated['contexts']

        # 5. Build diagnostics summary
        output.diagnostics = {
            'total_episodes': len(oc_input.episodes),
            'total_steps': len(all_steps),
            'curated_steps': len(output.curated_steps),
            'unique_contexts': len(output.context_coverage),
            'context_coverage': output.context_coverage,
            'bias_report': output.bias_report,
            'calibration': output.calibration_metrics
        }

        # 6. Quality gate decision
        gate_result = self._quality_gate(output)
        output.gate_status = gate_result['status']
        output.gate_reason = gate_result['reason']
        output.passes_quality_gate = gate_result['status'] in ['PASS', 'WARNING']

        return output

    def _analyze_coverage(self, steps: List[Dict]) -> Dict:
        """Analyze context coverage distribution."""
        context_counts = defaultdict(int)
        context_errors = defaultdict(list)
        context_uncertainties = defaultdict(list)

        for step in steps:
            ctx = step.get('context')
            if ctx is not None:
                ctx_key = tuple(ctx) if isinstance(ctx, (list, tuple)) else ctx
                context_counts[ctx_key] += 1
                context_errors[ctx_key].append(step.get('error', 0))
                context_uncertainties[ctx_key].append(step.get('uncertainty', 0))

        # Compute stats per context
        coverage = {}
        for ctx, count in context_counts.items():
            errors = context_errors[ctx]
            uncerts = context_uncertainties[ctx]
            coverage[str(ctx)] = {
                'count': count,
                'mean_error': float(np.mean(errors)) if errors else 0,
                'mean_uncertainty': float(np.mean(uncerts)) if uncerts else 0,
                'proportion': count / len(steps)
            }

        return coverage

    def _detect_biases(self, steps: List[Dict], episodes: List[Dict]) -> Dict:
        """Detect distribution biases in queried data."""
        # Context distribution
        context_counts = defaultdict(int)
        for step in steps:
            ctx = step.get('context')
            if ctx is not None:
                ctx_key = str(tuple(ctx) if isinstance(ctx, (list, tuple)) else ctx)
                context_counts[ctx_key] += 1

        total = len(steps)
        context_proportions = {k: v/total for k, v in context_counts.items()}

        # Check for severe imbalance
        max_proportion = max(context_proportions.values()) if context_proportions else 0
        imbalanced = max_proportion > 0.5  # More than 50% in one context

        # Analyze temporal bias (are later queries from different distribution?)
        early_contexts = defaultdict(int)
        late_contexts = defaultdict(int)
        midpoint = len(steps) // 2

        for i, step in enumerate(steps):
            ctx = step.get('context')
            if ctx is not None:
                ctx_key = str(tuple(ctx) if isinstance(ctx, (list, tuple)) else ctx)
                if i < midpoint:
                    early_contexts[ctx_key] += 1
                else:
                    late_contexts[ctx_key] += 1

        # Compute distribution shift
        all_contexts = set(early_contexts.keys()) | set(late_contexts.keys())
        distribution_shift = 0.0

        if all_contexts and len(steps) > 10:
            early_total = sum(early_contexts.values())
            late_total = sum(late_contexts.values())

            for ctx in all_contexts:
                early_prop = early_contexts.get(ctx, 0) / max(early_total, 1)
                late_prop = late_contexts.get(ctx, 0) / max(late_total, 1)
                distribution_shift += abs(early_prop - late_prop)

            distribution_shift /= len(all_contexts)

        return {
            'context_distribution': dict(context_counts),
            'context_proportions': context_proportions,
            'imbalanced': imbalanced,
            'max_proportion': max_proportion,
            'distribution_shift': distribution_shift,
            'temporal_drift_detected': distribution_shift > 0.3
        }

    def _analyze_calibration(self, steps: List[Dict]) -> Dict:
        """Analyze prediction calibration (uncertainty vs error)."""
        errors = []
        uncertainties = []
        predictions = []
        true_labels = []

        for step in steps:
            err = step.get('error')
            unc = step.get('uncertainty')
            pred = step.get('prediction')
            true = step.get('true_label')

            if all(v is not None and not np.isnan(v) for v in [err, unc, pred, true]):
                errors.append(err)
                uncertainties.append(unc)
                predictions.append(pred)
                true_labels.append(true)

        if len(errors) < 10:
            return {
                'uncertainty_error_correlation': 0.0,
                'mean_error': 0.0,
                'mean_uncertainty': 0.0,
                'coverage_1std': 0.0,
                'coverage_2std': 0.0,
                'sufficient_data': False
            }

        errors = np.array(errors)
        uncertainties = np.array(uncertainties)

        # Spearman correlation
        from scipy.stats import spearmanr
        corr, p_value = spearmanr(uncertainties, errors)

        # Coverage analysis
        coverage_1std = np.mean(errors <= uncertainties)
        coverage_2std = np.mean(errors <= 2 * uncertainties)

        return {
            'uncertainty_error_correlation': float(corr),
            'correlation_p_value': float(p_value),
            'mean_error': float(np.mean(errors)),
            'mean_uncertainty': float(np.mean(uncertainties)),
            'coverage_1std': float(coverage_1std),
            'coverage_2std': float(coverage_2std),
            'sufficient_data': True
        }

    def _curate_steps(self, steps: List[Dict]) -> Dict:
        """Filter and curate steps for FTB."""
        curated_steps = []
        smiles = []
        labels = []
        contexts = []

        for step in steps:
            # Filter by error threshold
            error = step.get('error', float('inf'))
            if error > self.max_error_threshold:
                continue

            # Extract data
            s = step.get('smiles')
            l = step.get('true_label')
            c = step.get('context')

            if s is not None and l is not None:
                curated_steps.append(step)
                smiles.append(s)
                labels.append(l)
                contexts.append(c)

        return {
            'steps': curated_steps,
            'smiles': smiles,
            'labels': labels,
            'contexts': contexts
        }

    def _quality_gate(self, output: MolecularOCOutput) -> Dict:
        """Determine if data quality is sufficient for FTB."""
        reasons = []

        # Check minimum curated data
        if len(output.curated_steps) < 20:
            return {
                'status': 'FAIL',
                'reason': f"Insufficient curated steps: {len(output.curated_steps)} < 20"
            }

        # Check calibration (optional warning)
        cal = output.calibration_metrics
        if cal.get('sufficient_data', False):
            corr = cal.get('uncertainty_error_correlation', 0)
            if corr < self.calibration_threshold:
                reasons.append(f"Poor calibration (corr={corr:.3f})")

        # Check context diversity
        n_contexts = len(output.context_coverage)
        if n_contexts < 3:
            reasons.append(f"Low context diversity ({n_contexts} contexts)")

        # Check bias
        if output.bias_report.get('imbalanced', False):
            reasons.append(f"Context imbalance (max={output.bias_report['max_proportion']:.1%})")

        if output.bias_report.get('temporal_drift_detected', False):
            reasons.append("Temporal distribution drift detected")

        # Final decision
        if len(reasons) == 0:
            return {'status': 'PASS', 'reason': 'All quality checks passed'}
        elif len(reasons) <= 2:
            return {'status': 'WARNING', 'reason': '; '.join(reasons)}
        else:
            return {'status': 'FAIL', 'reason': '; '.join(reasons)}

    def oc_output_to_ftb_format(self, oc_output: MolecularOCOutput) -> Optional[MolecularFTBInput]:
        """
        Convert OC output to FTB input format.

        Args:
            oc_output: Output from analyze()

        Returns:
            MolecularFTBInput or None if quality gate failed
        """
        if not oc_output.passes_quality_gate:
            return None

        # Compute quality weights based on uncertainty calibration
        weights = []
        for step in oc_output.curated_steps:
            # Higher weight for lower uncertainty predictions
            unc = step.get('uncertainty', 1.0)
            err = step.get('error', 1.0)

            # Weight = 1 / (1 + normalized_error)
            # Gives higher weight to accurate predictions
            weight = 1.0 / (1.0 + err)
            weights.append(weight)

        # Normalize weights
        if weights:
            max_weight = max(weights)
            weights = [w / max_weight for w in weights]

        return MolecularFTBInput(
            smiles=oc_output.curated_smiles,
            labels=oc_output.curated_labels,
            contexts=oc_output.curated_contexts,
            weights=weights,
            quality_metrics={
                'gate_status': oc_output.gate_status,
                'calibration': oc_output.calibration_metrics,
                'n_curated': len(oc_output.curated_steps)
            }
        )
