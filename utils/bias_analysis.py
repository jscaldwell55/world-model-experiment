"""
Bias Analysis for Molecular Property Predictions

Identifies systematic biases in model predictions based on molecular features.
Biases indicate regions of chemical space where the model consistently fails.

These biased regions are prime targets for synthetic data generation.
"""

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns


class BiasAnalyzer:
    """
    Analyze systematic biases in molecular property predictions.

    Identifies molecular features that correlate with high prediction errors,
    indicating model weaknesses that can be addressed with targeted synthetics.
    """

    def __init__(self, threshold_percentile: int = 75):
        """
        Initialize bias analyzer.

        Args:
            threshold_percentile: Percentile to split high/low feature values
                                (default: 75 = top 25% vs bottom 75%)
        """
        self.threshold_percentile = threshold_percentile

    def analyze_biases(self,
                      predictions_df: pd.DataFrame,
                      min_count: int = 15,
                      min_error_increase: float = 0.05) -> Dict[str, Dict]:
        """
        Identify systematic biases in predictions.

        Args:
            predictions_df: DataFrame with columns: smiles, error, prediction
            min_count: Minimum molecules in high-value group to flag bias
            min_error_increase: Minimum error increase to flag as bias

        Returns:
            Dict mapping feature_name -> bias_info
        """
        print(f"\nAnalyzing biases (threshold={self.threshold_percentile}th percentile)...")

        # Compute molecular features for all molecules
        features_df = self._compute_features(predictions_df['smiles'])

        # Merge with predictions
        data = pd.concat([predictions_df, features_df], axis=1)

        biases = {}

        # Check each feature for bias
        for feature in features_df.columns:
            bias_info = self._check_feature_bias(
                data,
                feature,
                min_count=min_count,
                min_error_increase=min_error_increase
            )

            if bias_info is not None:
                biases[feature] = bias_info
                print(f"  ✅ Found bias: {feature}")
                print(f"     Threshold: {bias_info['threshold']:.2f}")
                print(f"     Error increase: {bias_info['error_increase']:.3f}")
                print(f"     High count: {bias_info['count_high']}, Low count: {bias_info['count_low']}")

        if len(biases) == 0:
            print("  No significant biases found")

        return biases

    def _compute_features(self, smiles_list: List[str]) -> pd.DataFrame:
        """
        Compute molecular features for bias analysis.

        Args:
            smiles_list: List of SMILES strings

        Returns:
            DataFrame with molecular features
        """
        features = {
            'MW': [],
            'LogP': [],
            'TPSA': [],
            'NumHDonors': [],
            'NumHAcceptors': [],
            'NumRotatableBonds': [],
            'NumAromaticRings': [],
            'NumAliphaticRings': [],
            'FractionCSP3': []
        }

        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                # Fill with NaN for invalid molecules
                for key in features:
                    features[key].append(np.nan)
                continue

            features['MW'].append(Descriptors.MolWt(mol))
            features['LogP'].append(Descriptors.MolLogP(mol))
            features['TPSA'].append(Descriptors.TPSA(mol))
            features['NumHDonors'].append(Descriptors.NumHDonors(mol))
            features['NumHAcceptors'].append(Descriptors.NumHAcceptors(mol))
            features['NumRotatableBonds'].append(Descriptors.NumRotatableBonds(mol))
            features['NumAromaticRings'].append(Descriptors.NumAromaticRings(mol))
            features['NumAliphaticRings'].append(Descriptors.NumAliphaticRings(mol))
            features['FractionCSP3'].append(Descriptors.FractionCSP3(mol))

        return pd.DataFrame(features)

    def _check_feature_bias(self,
                           data: pd.DataFrame,
                           feature: str,
                           min_count: int = 15,
                           min_error_increase: float = 0.05) -> Dict:
        """
        Check if a specific feature shows systematic bias.

        Args:
            data: DataFrame with feature, error columns
            feature: Feature name to check
            min_count: Minimum count in high-value group
            min_error_increase: Minimum error increase to flag

        Returns:
            Bias info dict if bias found, None otherwise
        """
        # Remove NaN values
        valid_data = data[[feature, 'error']].dropna()

        if len(valid_data) < min_count * 2:
            return None

        # Split at threshold percentile
        threshold = valid_data[feature].quantile(self.threshold_percentile / 100)

        high_mask = valid_data[feature] >= threshold
        low_mask = valid_data[feature] < threshold

        high_errors = valid_data[high_mask]['error']
        low_errors = valid_data[low_mask]['error']

        # Check if high-value group has enough samples
        if len(high_errors) < min_count:
            return None

        # Compute mean errors
        high_error_mean = high_errors.mean()
        low_error_mean = low_errors.mean()

        error_increase = high_error_mean - low_error_mean

        # Flag as bias if error significantly increases
        if abs(error_increase) >= min_error_increase:
            # Compute bias direction (positive = overpredicts, negative = underpredicts)
            high_predictions = data.loc[valid_data.index[high_mask], 'prediction']
            high_actuals = data.loc[valid_data.index[high_mask], 'solubility']
            bias_direction = (high_predictions - high_actuals).mean()

            return {
                'threshold': threshold,
                'error_increase': error_increase,
                'high_error': high_error_mean,
                'low_error': low_error_mean,
                'count_high': len(high_errors),
                'count_low': len(low_errors),
                'bias_direction': bias_direction
            }

        return None

    def visualize_biases(self,
                        predictions_df: pd.DataFrame,
                        output_path: str):
        """
        Create visualization of prediction biases.

        Args:
            predictions_df: DataFrame with predictions
            output_path: Path to save visualization
        """
        print(f"\nGenerating bias visualization...")

        # Compute features
        features_df = self._compute_features(predictions_df['smiles'])
        data = pd.concat([predictions_df, features_df], axis=1)

        # Create figure with subplots
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        axes = axes.flatten()

        features = ['MW', 'LogP', 'TPSA', 'NumHDonors', 'NumHAcceptors',
                   'NumRotatableBonds', 'NumAromaticRings', 'NumAliphaticRings',
                   'FractionCSP3']

        for idx, feature in enumerate(features):
            ax = axes[idx]

            # Scatter plot: feature vs error
            valid_data = data[[feature, 'error']].dropna()

            if len(valid_data) > 0:
                ax.scatter(valid_data[feature], valid_data['error'], alpha=0.5, s=20)

                # Add threshold line
                threshold = valid_data[feature].quantile(self.threshold_percentile / 100)
                ax.axvline(threshold, color='r', linestyle='--',
                          label=f'{self.threshold_percentile}th percentile')

                # Add mean error lines
                high_mask = valid_data[feature] >= threshold
                if high_mask.sum() > 0:
                    high_error = valid_data[high_mask]['error'].mean()
                    ax.axhline(high_error, xmin=0.6, xmax=1.0,
                             color='orange', linestyle='-', linewidth=2,
                             label=f'High mean: {high_error:.2f}')

                low_mask = valid_data[feature] < threshold
                if low_mask.sum() > 0:
                    low_error = valid_data[low_mask]['error'].mean()
                    ax.axhline(low_error, xmin=0.0, xmax=0.6,
                             color='green', linestyle='-', linewidth=2,
                             label=f'Low mean: {low_error:.2f}')

                ax.set_xlabel(feature)
                ax.set_ylabel('Prediction Error')
                ax.set_title(f'{feature} vs Error')
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved visualization to {output_path}")


if __name__ == "__main__":
    # Test bias analyzer
    print("Testing bias analyzer...")

    # Create dummy data
    np.random.seed(42)
    n = 100
    dummy_data = pd.DataFrame({
        'smiles': ['C' * i for i in range(1, n+1)],  # Dummy SMILES
        'error': np.random.rand(n),
        'prediction': np.random.randn(n),
        'solubility': np.random.randn(n)
    })

    # Add bias: high MW molecules have higher errors
    mw_values = np.random.uniform(100, 500, n)
    dummy_data['error'] = dummy_data['error'] + 0.3 * (mw_values > 300)

    analyzer = BiasAnalyzer(threshold_percentile=75)

    # This won't work with dummy SMILES, but shows the interface
    print("Bias analyzer created successfully")
    print(f"Threshold percentile: {analyzer.threshold_percentile}")
