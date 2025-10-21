"""Statistical analysis for token-level prediction experiments.

This module implements comprehensive analyses (A1-A5) to test whether
linguistic next-token prediction encodes similar uncertainty signals as
grounded world-model prediction (belief surprisal).
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import precision_recall_curve, auc
from sklearn.linear_model import LinearRegression
import json
from pathlib import Path


class TokenAnalysis:
    """Analyze token prediction results and compare with belief surprisal.

    This class loads token logs and computes:
    - A1: Coupling (correlation between token NLL and belief surprisal)
    - A2: Surprise detection (does high token NLL predict surprisal spikes?)
    - A3: Predictive validity (does low NLL predict higher accuracy later?)
    - A4: Calibration (compare token vs belief calibration)
    - A5: Family factor (model-family effects)
    """

    def __init__(self, log_dir: str):
        """Initialize with directory containing token logs.

        Args:
            log_dir: Path to directory with *_token.json files
        """
        self.log_dir = log_dir
        self.df = self._load_all_logs()

    def _load_all_logs(self) -> pd.DataFrame:
        """Load all token logs into unified DataFrame.

        Returns:
            DataFrame with columns: episode_id, environment, agent_type, step,
            token_nll, per_token_nll, belief_surprisal, accuracy, num_tokens
        """
        records = []

        log_files = list(Path(self.log_dir).glob("*_token.json"))

        for log_file in log_files:
            with open(log_file) as f:
                data = json.load(f)

            episode_id = data['episode_id']

            # Parse environment and agent from episode_id
            # Format: "EnvironmentName_AgentName_ep000"
            parts = episode_id.rsplit('_', 1)[0].split('_')

            if len(parts) >= 2:
                agent_part = parts[-1].replace('Agent', '')
                env_part = '_'.join(parts[:-1]).replace('Lab', '')
            else:
                env_part = parts[0]
                agent_part = 'unknown'

            for entry in data.get('entries', []):
                records.append({
                    'episode_id': episode_id,
                    'environment': env_part,
                    'agent_type': agent_part,
                    'step': entry['step'],
                    'token_nll': entry['sequence_nll'],
                    'per_token_nll': entry['per_token_nll'],
                    'belief_surprisal': entry.get('belief_surprisal'),
                    'accuracy': entry.get('accuracy'),
                    'num_tokens': len(entry.get('tokens', [])),
                })

        return pd.DataFrame(records)

    # === A1: COUPLING ANALYSIS ===

    def compute_coupling(self, method: str = 'pearson') -> pd.DataFrame:
        """A1: Compute correlation between token NLL and belief surprisal.

        Args:
            method: 'pearson' or 'spearman' (currently computes both)

        Returns:
            DataFrame with correlation results by environment
        """
        df_valid = self.df.dropna(subset=['token_nll', 'belief_surprisal'])

        if len(df_valid) == 0:
            return pd.DataFrame()

        results = []

        for env in sorted(df_valid['environment'].unique()):
            env_data = df_valid[df_valid['environment'] == env]

            if len(env_data) < 3:
                continue

            # Pearson correlation
            r_p, p_p = pearsonr(env_data['token_nll'], env_data['belief_surprisal'])

            # Spearman correlation
            r_s, p_s = spearmanr(env_data['token_nll'], env_data['belief_surprisal'])

            results.append({
                'environment': env,
                'pearson_r': r_p,
                'pearson_p': p_p,
                'spearman_r': r_s,
                'spearman_p': p_s,
                'n_steps': len(env_data),
                'mean_token_nll': env_data['token_nll'].mean(),
                'std_token_nll': env_data['token_nll'].std(),
                'mean_belief_surprisal': env_data['belief_surprisal'].mean(),
                'std_belief_surprisal': env_data['belief_surprisal'].std(),
            })

        return pd.DataFrame(results)

    def compute_coupling_by_agent(self) -> pd.DataFrame:
        """A1: Coupling stratified by agent type.

        Returns:
            DataFrame with coupling results for each environment × agent combination
        """
        df_valid = self.df.dropna(subset=['token_nll', 'belief_surprisal'])

        results = []

        for env in sorted(df_valid['environment'].unique()):
            for agent in sorted(df_valid['agent_type'].unique()):
                data = df_valid[
                    (df_valid['environment'] == env) &
                    (df_valid['agent_type'] == agent)
                ]

                if len(data) < 3:
                    continue

                r, p = pearsonr(data['token_nll'], data['belief_surprisal'])

                results.append({
                    'environment': env,
                    'agent_type': agent,
                    'pearson_r': r,
                    'p_value': p,
                    'n_steps': len(data)
                })

        return pd.DataFrame(results)

    # === A2: SURPRISE DETECTION ===

    def compute_surprise_detection(
        self,
        surprisal_threshold: float = 2.0
    ) -> pd.DataFrame:
        """A2: Test if high token NLL predicts high-surprisal events.

        Uses precision-recall AUC to evaluate how well token NLL detects
        belief surprisal spikes.

        Args:
            surprisal_threshold: Threshold for defining "high surprisal"

        Returns:
            DataFrame with PR-AUC by environment
        """
        df_valid = self.df.dropna(subset=['token_nll', 'belief_surprisal'])

        results = []

        for env in sorted(df_valid['environment'].unique()):
            env_data = df_valid[df_valid['environment'] == env].copy()

            if len(env_data) < 5:
                continue

            # Label high-surprisal events
            env_data['high_surprisal'] = (
                env_data['belief_surprisal'] > surprisal_threshold
            ).astype(int)

            num_high = env_data['high_surprisal'].sum()

            if num_high == 0 or num_high == len(env_data):
                # All same label - skip
                continue

            # Compute PR curve
            precision, recall, _ = precision_recall_curve(
                env_data['high_surprisal'],
                env_data['token_nll']
            )

            pr_auc = auc(recall, precision)

            results.append({
                'environment': env,
                'pr_auc': pr_auc,
                'threshold': surprisal_threshold,
                'num_high_surprisal': num_high,
                'total_steps': len(env_data),
                'prevalence': num_high / len(env_data)
            })

        return pd.DataFrame(results)

    # === A3: PREDICTIVE VALIDITY ===

    def compute_predictive_validity(self, lag: int = 1) -> pd.DataFrame:
        """A3: Test if low token NLL at step t predicts high accuracy at t+lag.

        Args:
            lag: Number of steps to look ahead

        Returns:
            DataFrame with regression results per episode
        """
        results = []

        for episode_id in self.df['episode_id'].unique():
            ep_data = self.df[self.df['episode_id'] == episode_id].copy()
            ep_data = ep_data.sort_values('step')

            # Shift accuracy by lag
            ep_data['accuracy_future'] = ep_data['accuracy'].shift(-lag)

            # Get valid data
            valid_data = ep_data.dropna(subset=['token_nll', 'accuracy_future'])

            if len(valid_data) < 3:
                continue

            # Correlate token NLL with future accuracy
            r, p = pearsonr(valid_data['token_nll'], valid_data['accuracy_future'])

            # Also fit linear regression
            X = valid_data[['token_nll']].values
            y = valid_data['accuracy_future'].values

            model = LinearRegression()
            model.fit(X, y)

            results.append({
                'episode_id': episode_id,
                'environment': ep_data['environment'].iloc[0],
                'agent_type': ep_data['agent_type'].iloc[0],
                'correlation': r,
                'p_value': p,
                'regression_coef': model.coef_[0],
                'regression_intercept': model.intercept_,
                'lag': lag,
                'n_steps': len(valid_data)
            })

        return pd.DataFrame(results)

    # === A4: CALIBRATION COMPARISON ===

    def compute_token_calibration(self) -> Dict[str, float]:
        """A4: Compute calibration metrics for token predictions.

        Converts token NLL to pseudo-confidence and computes Brier score
        and Expected Calibration Error (ECE).

        Returns:
            Dictionary with calibration metrics
        """
        df_valid = self.df.dropna(subset=['token_nll', 'accuracy'])

        if len(df_valid) == 0:
            return {}

        # Convert NLL to confidence (lower NLL → higher confidence)
        # Use sigmoid transformation: conf = 1 / (1 + exp(nll - median))
        median_nll = df_valid['token_nll'].median()
        df_valid = df_valid.copy()
        df_valid['token_confidence'] = 1 / (1 + np.exp(df_valid['token_nll'] - median_nll))

        # Brier score: mean((confidence - actual)^2)
        brier = ((df_valid['token_confidence'] - df_valid['accuracy']) ** 2).mean()

        # Expected Calibration Error (ECE)
        # Bin predictions and compute |mean_conf - mean_acc| per bin
        num_bins = 10
        df_valid['conf_bin'] = pd.cut(
            df_valid['token_confidence'],
            bins=num_bins,
            labels=False
        )

        ece = 0.0
        for bin_idx in range(num_bins):
            bin_data = df_valid[df_valid['conf_bin'] == bin_idx]
            if len(bin_data) > 0:
                mean_conf = bin_data['token_confidence'].mean()
                mean_acc = bin_data['accuracy'].mean()
                weight = len(bin_data) / len(df_valid)
                ece += weight * abs(mean_conf - mean_acc)

        return {
            'brier_score': float(brier),
            'expected_calibration_error': float(ece),
            'n_samples': len(df_valid),
            'mean_confidence': float(df_valid['token_confidence'].mean()),
            'mean_accuracy': float(df_valid['accuracy'].mean())
        }

    # === A5: FAMILY FACTOR ===

    def compute_family_effects(self) -> pd.DataFrame:
        """A5: Analyze model-family × environment interactions.

        Note: This requires extending the log format to include model_family.
        Currently returns placeholder indicating this feature needs implementation.

        Returns:
            DataFrame with interaction effects (placeholder for now)
        """
        # NOTE: This requires extending the log format to include model_family
        # For now, return placeholder
        return pd.DataFrame({
            'note': ['Requires model_family tracking in token logs'],
            'implementation': ['Add model_family field to TokenLogEntry'],
            'usage': ['Can stratify coupling by (GPT-4 vs Claude) × environment']
        })

    # === SUMMARY REPORT ===

    def generate_summary_report(self) -> str:
        """Generate comprehensive text summary of all analyses.

        Returns:
            Multi-line string with formatted report
        """

        report = []
        report.append("=" * 70)
        report.append("TOKEN PREDICTION ANALYSIS REPORT")
        report.append("=" * 70)
        report.append("")

        # Dataset summary
        report.append("DATASET SUMMARY")
        report.append("-" * 70)
        report.append(f"Total steps: {len(self.df)}")
        report.append(f"Episodes: {self.df['episode_id'].nunique()}")
        report.append(f"Environments: {', '.join(sorted(self.df['environment'].unique()))}")
        report.append(f"Agent types: {', '.join(sorted(self.df['agent_type'].unique()))}")
        report.append("")

        # A1: Coupling
        report.append("A1: COUPLING ANALYSIS")
        report.append("-" * 70)
        coupling = self.compute_coupling()
        if len(coupling) > 0:
            for _, row in coupling.iterrows():
                report.append(
                    f"{row['environment']:15s}: r={row['pearson_r']:+.3f} "
                    f"(p={row['pearson_p']:.4f}), n={row['n_steps']}"
                )
        else:
            report.append("No valid data for coupling analysis")
        report.append("")

        # A2: Surprise detection
        report.append("A2: SURPRISE DETECTION")
        report.append("-" * 70)
        surprise = self.compute_surprise_detection()
        if len(surprise) > 0:
            for _, row in surprise.iterrows():
                report.append(
                    f"{row['environment']:15s}: PR-AUC={row['pr_auc']:.3f}, "
                    f"prevalence={row['prevalence']:.2%}"
                )
        else:
            report.append("No valid data for surprise detection")
        report.append("")

        # A3: Predictive validity
        report.append("A3: PREDICTIVE VALIDITY")
        report.append("-" * 70)
        validity = self.compute_predictive_validity()
        if len(validity) > 0:
            summary = validity.groupby('environment')['correlation'].agg(['mean', 'std', 'count'])
            for env, row in summary.iterrows():
                report.append(
                    f"{env:15s}: mean_r={row['mean']:+.3f} "
                    f"(std={row['std']:.3f}), n={int(row['count'])}"
                )
        else:
            report.append("No valid data for predictive validity")
        report.append("")

        # A4: Calibration
        report.append("A4: CALIBRATION")
        report.append("-" * 70)
        calib = self.compute_token_calibration()
        if calib:
            report.append(f"Brier Score: {calib['brier_score']:.4f}")
            report.append(f"Expected Calibration Error: {calib['expected_calibration_error']:.4f}")
            report.append(f"Mean Confidence: {calib['mean_confidence']:.4f}")
            report.append(f"Mean Accuracy: {calib['mean_accuracy']:.4f}")
        else:
            report.append("No valid data for calibration")
        report.append("")

        report.append("=" * 70)

        return "\n".join(report)
