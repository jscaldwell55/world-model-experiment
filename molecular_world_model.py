"""
MolecularWorldModel: Context-aware world model for molecular property prediction.

This module implements a world model designed for agentic molecular property prediction,
supporting Offline Consolidation (OC) architecture. Unlike single-step regression models,
this model is designed to:
1. Predict properties with calibrated uncertainty
2. Support context-aware predictions (scaffold, MW, LogP bins)
3. Enable incremental updates for Fine-Tuning Bridge (FTB) integration
4. Provide calibration metrics for evaluating prediction quality

Key insight: OC is designed for consolidating TRAJECTORIES from agentic decision-making,
not single-step SMILES→property regression.
"""

import pickle
import warnings
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# RDKit imports
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import DataStructs

# Suppress RDKit deprecation warnings
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


class MolecularWorldModel:
    """
    Context-aware world model for molecular property prediction.

    Key features:
    - Predicts property + uncertainty using Random Forest ensemble
    - Context-aware: conditions on (scaffold_cluster, MW_bin, LogP_bin)
    - Supports incremental updates (for FTB integration)
    - Provides calibration metrics for uncertainty evaluation

    Attributes:
        n_scaffold_clusters: Number of clusters for scaffold grouping
        n_mw_bins: Number of bins for molecular weight
        n_logp_bins: Number of bins for LogP
        n_estimators: Number of trees in Random Forest
    """

    def __init__(self,
                 n_scaffold_clusters: int = 20,
                 n_mw_bins: int = 5,
                 n_logp_bins: int = 5,
                 n_estimators: int = 100,
                 random_state: int = 42):
        """
        Initialize MolecularWorldModel.

        Args:
            n_scaffold_clusters: Number of scaffold clusters for context
            n_mw_bins: Number of molecular weight bins
            n_logp_bins: Number of LogP bins
            n_estimators: Number of trees in RF ensemble
            random_state: Random seed for reproducibility
        """
        self.n_scaffold_clusters = n_scaffold_clusters
        self.n_mw_bins = n_mw_bins
        self.n_logp_bins = n_logp_bins
        self.n_estimators = n_estimators
        self.random_state = random_state

        # Model components
        self.rf_model = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1
        )

        # Context components (initialized during fit)
        self.scaffold_clusterer = None
        self.scaffold_to_cluster = {}
        self.scaffold_fps = {}  # Cache scaffold fingerprints
        self.cluster_centroids = None

        # Binning thresholds
        self.mw_bins = None
        self.logp_bins = None

        # Training data cache (for incremental updates)
        self._train_smiles = []
        self._train_y = []
        self._train_fps = []

        # Fitted flag
        self.is_fitted = False

    def _compute_morgan_fp(self, mol_or_smiles, radius: int = 2, n_bits: int = 1024) -> Optional[np.ndarray]:
        """Compute Morgan fingerprint as numpy array."""
        if isinstance(mol_or_smiles, str):
            mol = Chem.MolFromSmiles(mol_or_smiles)
        else:
            mol = mol_or_smiles

        if mol is None:
            return None

        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        arr = np.zeros(n_bits, dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr

    def _compute_descriptors(self, mol_or_smiles) -> Optional[Dict[str, float]]:
        """Compute RDKit descriptors."""
        if isinstance(mol_or_smiles, str):
            mol = Chem.MolFromSmiles(mol_or_smiles)
        else:
            mol = mol_or_smiles

        if mol is None:
            return None

        return {
            'MolWt': Descriptors.MolWt(mol),
            'LogP': Descriptors.MolLogP(mol),
            'NumHDonors': Descriptors.NumHDonors(mol),
            'NumHAcceptors': Descriptors.NumHAcceptors(mol),
            'TPSA': Descriptors.TPSA(mol),
            'NumRotatableBonds': Descriptors.NumRotatableBonds(mol),
            'NumAromaticRings': rdMolDescriptors.CalcNumAromaticRings(mol)
        }

    def _get_murcko_scaffold(self, mol_or_smiles) -> Optional[str]:
        """Get Murcko scaffold SMILES."""
        if isinstance(mol_or_smiles, str):
            mol = Chem.MolFromSmiles(mol_or_smiles)
        else:
            mol = mol_or_smiles

        if mol is None:
            return None

        try:
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            return Chem.MolToSmiles(scaffold)
        except:
            return None

    def _compute_features(self, smiles: str) -> Optional[Tuple[np.ndarray, Dict[str, float], str]]:
        """Compute all features for a molecule."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        fp = self._compute_morgan_fp(mol)
        descriptors = self._compute_descriptors(mol)
        scaffold = self._get_murcko_scaffold(mol)

        if fp is None or descriptors is None:
            return None

        return fp, descriptors, scaffold

    def _build_feature_matrix(self, smiles_list: List[str]) -> Tuple[np.ndarray, List[Dict], List[str]]:
        """Build feature matrix from SMILES list."""
        fps = []
        descriptors_list = []
        scaffolds = []
        valid_indices = []

        for i, smiles in enumerate(smiles_list):
            result = self._compute_features(smiles)
            if result is not None:
                fp, desc, scaffold = result
                fps.append(fp)
                descriptors_list.append(desc)
                scaffolds.append(scaffold)
                valid_indices.append(i)

        if not fps:
            raise ValueError("No valid molecules found")

        # Combine FP and descriptors into feature matrix
        fp_matrix = np.array(fps)
        desc_matrix = np.array([
            [d['MolWt'], d['LogP'], d['NumHDonors'], d['NumHAcceptors'],
             d['TPSA'], d['NumRotatableBonds'], d['NumAromaticRings']]
            for d in descriptors_list
        ])

        X = np.hstack([fp_matrix, desc_matrix])

        return X, descriptors_list, scaffolds, valid_indices

    def _cluster_scaffolds(self, scaffolds: List[str]) -> None:
        """Cluster scaffolds using Tanimoto similarity on Morgan FPs."""
        unique_scaffolds = list(set([s for s in scaffolds if s is not None]))

        if len(unique_scaffolds) <= self.n_scaffold_clusters:
            # If fewer unique scaffolds than clusters, use direct mapping
            self.scaffold_to_cluster = {s: i for i, s in enumerate(unique_scaffolds)}
            self.scaffold_to_cluster[None] = len(unique_scaffolds)  # Unknown scaffold
            return

        # Compute fingerprints for scaffolds
        scaffold_fps = []
        valid_scaffolds = []

        for scaffold in unique_scaffolds:
            fp = self._compute_morgan_fp(scaffold)
            if fp is not None:
                scaffold_fps.append(fp)
                valid_scaffolds.append(scaffold)
                self.scaffold_fps[scaffold] = fp

        if len(scaffold_fps) < 2:
            self.scaffold_to_cluster = {s: 0 for s in unique_scaffolds}
            self.scaffold_to_cluster[None] = 0
            return

        # Compute distance matrix (1 - Tanimoto similarity)
        fp_matrix = np.array(scaffold_fps)
        n = len(fp_matrix)

        # Use precomputed distance matrix
        distances = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                # Tanimoto similarity
                intersection = np.sum(fp_matrix[i] * fp_matrix[j])
                union = np.sum(fp_matrix[i]) + np.sum(fp_matrix[j]) - intersection
                if union > 0:
                    sim = intersection / union
                else:
                    sim = 0
                distances[i, j] = 1 - sim
                distances[j, i] = distances[i, j]

        # Cluster using agglomerative clustering
        n_clusters = min(self.n_scaffold_clusters, len(valid_scaffolds))
        self.scaffold_clusterer = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric='precomputed',
            linkage='average'
        )

        labels = self.scaffold_clusterer.fit_predict(distances)

        # Build mapping
        for scaffold, label in zip(valid_scaffolds, labels):
            self.scaffold_to_cluster[scaffold] = int(label)

        # Unknown scaffold cluster
        self.scaffold_to_cluster[None] = n_clusters

        # Compute cluster centroids for assigning new scaffolds
        self.cluster_centroids = {}
        for cluster_id in range(n_clusters):
            cluster_fps = [fp_matrix[i] for i, l in enumerate(labels) if l == cluster_id]
            if cluster_fps:
                self.cluster_centroids[cluster_id] = np.mean(cluster_fps, axis=0)

    def _get_scaffold_cluster(self, scaffold: Optional[str]) -> int:
        """Get cluster ID for a scaffold."""
        if scaffold in self.scaffold_to_cluster:
            return self.scaffold_to_cluster[scaffold]

        # For unseen scaffolds, find nearest cluster
        if scaffold is None or self.cluster_centroids is None:
            return self.scaffold_to_cluster.get(None, 0)

        fp = self._compute_morgan_fp(scaffold)
        if fp is None:
            return self.scaffold_to_cluster.get(None, 0)

        # Find nearest cluster centroid
        best_cluster = 0
        best_sim = -1

        for cluster_id, centroid in self.cluster_centroids.items():
            intersection = np.sum(fp * centroid)
            union = np.sum(fp) + np.sum(centroid) - intersection
            if union > 0:
                sim = intersection / union
                if sim > best_sim:
                    best_sim = sim
                    best_cluster = cluster_id

        return best_cluster

    def _compute_bins(self, descriptors_list: List[Dict]) -> None:
        """Compute MW and LogP bins from training data."""
        mw_values = [d['MolWt'] for d in descriptors_list]
        logp_values = [d['LogP'] for d in descriptors_list]

        # Use quantile-based binning
        self.mw_bins = np.percentile(mw_values, np.linspace(0, 100, self.n_mw_bins + 1))
        self.logp_bins = np.percentile(logp_values, np.linspace(0, 100, self.n_logp_bins + 1))

    def _get_bin(self, value: float, bins: np.ndarray) -> int:
        """Get bin index for a value."""
        return min(np.searchsorted(bins[1:-1], value), len(bins) - 2)

    def fit(self,
            smiles_list: List[str],
            y: Union[List[float], np.ndarray],
            contexts: Optional[List[Tuple]] = None,
            sample_weight: Optional[List[float]] = None) -> 'MolecularWorldModel':
        """
        Train the world model on labeled molecules.

        Args:
            smiles_list: List of SMILES strings
            y: Target property values
            contexts: Optional pre-computed contexts (ignored, computed internally)
            sample_weight: Optional sample weights for weighted training

        Returns:
            self
        """
        y = np.array(y)
        if sample_weight is not None:
            sample_weight = np.array(sample_weight)

        # Build features
        X, descriptors_list, scaffolds, valid_indices = self._build_feature_matrix(smiles_list)
        y_valid = y[valid_indices]

        # Filter sample weights to valid indices
        if sample_weight is not None:
            weight_valid = sample_weight[valid_indices]
        else:
            weight_valid = None

        # Store training data for incremental updates
        self._train_smiles = [smiles_list[i] for i in valid_indices]
        self._train_y = y_valid.tolist()
        self._train_fps = X.tolist()

        # Cluster scaffolds
        self._cluster_scaffolds(scaffolds)

        # Compute bins
        self._compute_bins(descriptors_list)

        # Train RF model with optional sample weights
        self.rf_model.fit(X, y_valid, sample_weight=weight_valid)

        self.is_fitted = True

        return self

    def predict(self,
                smiles_list: Union[str, List[str]],
                return_uncertainty: bool = True) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        """
        Predict property values with optional uncertainty.

        Args:
            smiles_list: SMILES string or list of SMILES
            return_uncertainty: Whether to return uncertainty estimates

        Returns:
            If return_uncertainty=True: (predictions, uncertainties)
            Otherwise: predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        if isinstance(smiles_list, str):
            smiles_list = [smiles_list]
            single_input = True
        else:
            single_input = False

        # Build features
        try:
            X, _, _, valid_indices = self._build_feature_matrix(smiles_list)
        except ValueError:
            # No valid molecules
            n = len(smiles_list)
            preds = np.full(n, np.nan)
            uncerts = np.full(n, np.nan)
            if return_uncertainty:
                return preds, uncerts
            return preds

        # Get predictions from all trees
        all_preds = np.array([tree.predict(X) for tree in self.rf_model.estimators_])

        # Mean prediction
        mean_preds = np.mean(all_preds, axis=0)

        # Uncertainty as std of tree predictions
        uncertainties = np.std(all_preds, axis=0)

        # Fill in invalid molecules with NaN
        full_preds = np.full(len(smiles_list), np.nan)
        full_uncerts = np.full(len(smiles_list), np.nan)

        for i, idx in enumerate(valid_indices):
            full_preds[idx] = mean_preds[i]
            full_uncerts[idx] = uncertainties[i]

        if single_input:
            full_preds = full_preds[0]
            full_uncerts = full_uncerts[0]

        if return_uncertainty:
            return full_preds, full_uncerts
        return full_preds

    def get_context(self, smiles: str) -> Tuple[int, int, int]:
        """
        Get context tuple for a molecule.

        Args:
            smiles: SMILES string

        Returns:
            Tuple of (scaffold_cluster, mw_bin, logp_bin)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting context")

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return (self.n_scaffold_clusters, self.n_mw_bins - 1, self.n_logp_bins - 1)

        # Get scaffold cluster
        scaffold = self._get_murcko_scaffold(mol)
        scaffold_cluster = self._get_scaffold_cluster(scaffold)

        # Get descriptor bins
        desc = self._compute_descriptors(mol)
        mw_bin = self._get_bin(desc['MolWt'], self.mw_bins)
        logp_bin = self._get_bin(desc['LogP'], self.logp_bins)

        return (scaffold_cluster, mw_bin, logp_bin)

    def get_contexts(self, smiles_list: List[str]) -> List[Tuple[int, int, int]]:
        """Get contexts for multiple molecules."""
        return [self.get_context(s) for s in smiles_list]

    def update(self,
               new_smiles: Union[str, List[str]],
               new_y: Union[float, List[float]]) -> 'MolecularWorldModel':
        """
        Incremental update with new data.

        For now, this performs a full retrain including new data.
        Future versions could use incremental RF methods.

        Args:
            new_smiles: New SMILES string(s)
            new_y: New target value(s)

        Returns:
            self
        """
        if isinstance(new_smiles, str):
            new_smiles = [new_smiles]
            new_y = [new_y]

        # Combine with existing training data
        all_smiles = self._train_smiles + list(new_smiles)
        all_y = self._train_y + list(new_y)

        # Retrain
        return self.fit(all_smiles, all_y)

    def get_calibration_metrics(self,
                                smiles_list: List[str],
                                y_true: Union[List[float], np.ndarray]) -> Dict:
        """
        Compute calibration metrics for predictions.

        Args:
            smiles_list: Test SMILES
            y_true: True property values

        Returns:
            Dict with calibration metrics:
            - expected_calibration_error: ECE for uncertainty bins
            - uncertainty_error_correlation: Spearman correlation
            - coverage_by_context: Coverage stats per context
            - mae: Mean absolute error
            - rmse: Root mean squared error
            - r2: R-squared
        """
        y_true = np.array(y_true)

        # Get predictions and uncertainties
        preds, uncerts = self.predict(smiles_list, return_uncertainty=True)

        # Filter out invalid predictions
        valid_mask = ~np.isnan(preds)
        preds = preds[valid_mask]
        uncerts = uncerts[valid_mask]
        y_true = y_true[valid_mask]
        smiles_list = [s for s, v in zip(smiles_list, valid_mask) if v]

        if len(preds) == 0:
            return {
                'expected_calibration_error': np.nan,
                'uncertainty_error_correlation': np.nan,
                'coverage_by_context': {},
                'mae': np.nan,
                'rmse': np.nan,
                'r2': np.nan
            }

        # Basic metrics
        errors = np.abs(preds - y_true)
        mae = mean_absolute_error(y_true, preds)
        rmse = np.sqrt(mean_squared_error(y_true, preds))
        r2 = r2_score(y_true, preds)

        # Uncertainty-error correlation (Spearman)
        from scipy.stats import spearmanr
        corr, _ = spearmanr(uncerts, errors)

        # Expected Calibration Error
        # Bin by uncertainty and check if errors are calibrated
        n_bins = 10
        ece = 0.0
        uncert_bins = np.percentile(uncerts, np.linspace(0, 100, n_bins + 1))

        for i in range(n_bins):
            mask = (uncerts >= uncert_bins[i]) & (uncerts < uncert_bins[i+1])
            if np.sum(mask) > 0:
                bin_uncert = np.mean(uncerts[mask])
                bin_error = np.mean(errors[mask])
                # Ideal: mean error ≈ mean uncertainty (for Gaussian calibration)
                ece += np.abs(bin_error - bin_uncert) * np.sum(mask) / len(uncerts)

        # Coverage by context
        contexts = self.get_contexts(smiles_list)
        context_stats = {}

        for ctx in set(contexts):
            mask = [c == ctx for c in contexts]
            if sum(mask) > 5:  # Only report contexts with sufficient data
                ctx_errors = errors[mask]
                ctx_uncerts = uncerts[mask]
                ctx_preds = preds[mask]
                ctx_y = y_true[mask]

                context_stats[str(ctx)] = {
                    'n_samples': sum(mask),
                    'mae': float(np.mean(ctx_errors)),
                    'mean_uncertainty': float(np.mean(ctx_uncerts)),
                    'coverage_1std': float(np.mean(ctx_errors <= ctx_uncerts)),
                    'coverage_2std': float(np.mean(ctx_errors <= 2 * ctx_uncerts))
                }

        return {
            'expected_calibration_error': float(ece),
            'uncertainty_error_correlation': float(corr),
            'coverage_by_context': context_stats,
            'mae': float(mae),
            'rmse': float(rmse),
            'r2': float(r2),
            'n_samples': len(preds)
        }

    def save(self, filepath: str) -> None:
        """Save model to file."""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath: str) -> 'MolecularWorldModel':
        """Load model from file."""
        with open(filepath, 'rb') as f:
            return pickle.load(f)

    def get_feature_importances(self) -> Dict[str, float]:
        """Get feature importances from RF model."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted")

        importances = self.rf_model.feature_importances_

        # Feature names
        fp_names = [f'fp_{i}' for i in range(1024)]
        desc_names = ['MolWt', 'LogP', 'NumHDonors', 'NumHAcceptors',
                      'TPSA', 'NumRotatableBonds', 'NumAromaticRings']

        feature_names = fp_names + desc_names

        # Return sorted by importance
        imp_dict = dict(zip(feature_names, importances))
        return dict(sorted(imp_dict.items(), key=lambda x: -x[1]))

    def get_descriptor_importances(self) -> Dict[str, float]:
        """Get importances for descriptor features only (aggregating FP importance)."""
        importances = self.get_feature_importances()

        # Sum fingerprint importance
        fp_importance = sum(v for k, v in importances.items() if k.startswith('fp_'))

        # Get descriptor importances
        desc_imp = {k: v for k, v in importances.items() if not k.startswith('fp_')}
        desc_imp['MorganFP'] = fp_importance

        return dict(sorted(desc_imp.items(), key=lambda x: -x[1]))


def load_esol_data(filepath: str = 'data/esol_processed.pkl') -> Dict:
    """Load processed ESOL data."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)
