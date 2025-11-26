"""
File: agents/molecular_world_model.py

Purpose: Context-aware world model for molecular solubility prediction.
         Learns from episodes where agent observes (SMILES, solubility) pairs.

Architecture:
- Uses RandomForest for non-linear property prediction
- Maintains separate RF model per context (4 total)
- Trains incrementally as more molecules are observed
- Provides uncertainty estimates from tree variance

Interface:
- update_belief(smiles, observed_property) - Learn from observation
- predict_property(smiles) - Predict with uncertainty
- get_context(smiles) - Extract context for molecule

Expected workflow:
1. Initialize model
2. For each molecule in ESOL:
   - Extract context (aromatic/aliphatic, small/large)
   - Featurize (Morgan FP + descriptors)
   - Store in context-specific belief state
   - Train RF when context has 50+ samples
3. Predict solubility for new molecules
4. Use predictions for ACE agent decision-making
"""

from utils.context_spec_molecular import MOLECULAR_CONTEXT, extract_molecular_context
from sklearn.ensemble import RandomForestRegressor
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
import numpy as np
import pickle
import os
import warnings


class MolecularWorldModel:
    """
    World model for molecular property prediction using context-aware RandomForests.

    Attributes:
        context_spec: MOLECULAR_CONTEXT from utils/context_spec_molecular.py
        context_models: Dict mapping context tuple -> RandomForestRegressor
        context_counts: Dict mapping context tuple -> int (number of samples)
        belief_state: Dict mapping context tuple -> {'X': list, 'y': list}
                      Stores training data per context

    Parameters:
        n_estimators: Number of trees in RandomForest (default: 100)
        max_depth: Maximum tree depth (default: 10)
        min_samples_context: Minimum samples required to train context model (default: 50)
    """

    def __init__(self, n_estimators=100, max_depth=10, min_samples_context=50):
        """
        Initialize molecular world model.

        Args:
            n_estimators: Number of RF trees
            max_depth: Max depth per tree
            min_samples_context: Min samples to train context model (validated: 50)
        """
        # Use validated 4-context scheme
        self.context_spec = MOLECULAR_CONTEXT

        # Context-specific models
        self.context_models = {}   # {context: RandomForestRegressor}
        self.context_counts = {}   # {context: int}

        # Belief state: stores training data per context
        self.belief_state = {}  # {context: {'X': list, 'y': list}}

        # Hyperparameters (validated in Test 6)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_context = min_samples_context

    def featurize(self, smiles):
        """
        Convert SMILES to molecular feature vector.

        Features (1031 total):
        - Morgan fingerprint (ECFP4): 1024 bits, radius=2
        - Physicochemical descriptors: 7 values
          1. Molecular Weight (MW)
          2. LogP (partition coefficient)
          3. TPSA (topological polar surface area)
          4. HBD (hydrogen bond donors)
          5. HBA (hydrogen bond acceptors)
          6. NumRotatableBonds
          7. NumAromaticRings

        Args:
            smiles: SMILES string

        Returns:
            np.array of shape (1031,)

        Raises:
            ValueError: If SMILES is invalid
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")

        # Morgan fingerprint (ECFP4, radius=2, 1024 bits)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
        fp_array = np.array(fp)

        # Physicochemical descriptors
        descriptors = np.array([
            Descriptors.MolWt(mol),              # MW
            Descriptors.MolLogP(mol),            # LogP
            Descriptors.TPSA(mol),               # TPSA
            Descriptors.NumHDonors(mol),         # HBD
            Descriptors.NumHAcceptors(mol),      # HBA
            Descriptors.NumRotatableBonds(mol),  # Rotatable bonds
            Descriptors.NumAromaticRings(mol)    # Aromatic rings
        ])

        # Concatenate: 1024 + 7 = 1031 features
        features = np.concatenate([fp_array, descriptors])

        return features

    def get_context(self, smiles):
        """
        Extract context from SMILES string.

        Args:
            smiles: SMILES string

        Returns:
            Tuple: (aromatic_type, size_bin)
            Examples: ('aromatic', 'small'), ('aliphatic', 'large')
        """
        observation = {'smiles': smiles}
        return extract_molecular_context(observation)

    def update_belief(self, smiles, observed_property):
        """
        Learn from molecule-property observation (online learning).

        Workflow:
        1. Extract context
        2. Featurize molecule
        3. Store in context-specific belief state
        4. Increment context counter
        5. If context has ≥50 samples, train RF model

        Args:
            smiles: SMILES string
            observed_property: Measured solubility (logS units)
        """
        # Extract context
        context = self.get_context(smiles)

        # Skip invalid/unknown contexts
        if context[0] in ('invalid', 'unknown'):
            warnings.warn(f"Skipping molecule with context {context}: {smiles}")
            return

        # Featurize
        try:
            features = self.featurize(smiles)
        except ValueError as e:
            warnings.warn(f"Could not featurize {smiles}: {e}")
            return

        # Initialize context storage if needed
        if context not in self.belief_state:
            self.belief_state[context] = {
                'X': [],
                'y': []
            }

        # Store observation
        self.belief_state[context]['X'].append(features)
        self.belief_state[context]['y'].append(observed_property)

        # Update count
        self.context_counts[context] = self.context_counts.get(context, 0) + 1

        # Train model if enough samples (threshold: 50, validated in Test 6)
        if self.context_counts[context] >= self.min_samples_context:
            # Only train if we haven't already or if we've added significant new data
            if context not in self.context_models or \
               self.context_counts[context] % 25 == 0:  # Retrain every 25 samples
                self._train_context_model(context)

    def _train_context_model(self, context):
        """
        Train RandomForest model for specific context.

        Args:
            context: Tuple (aromatic_type, size_bin)
        """
        # Get training data
        X = np.array(self.belief_state[context]['X'])
        y = np.array(self.belief_state[context]['y'])

        # Train RandomForest
        model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=42,
            n_jobs=-1  # Use all CPU cores
        )
        model.fit(X, y)

        # Store model
        self.context_models[context] = model

        print(f"Trained RF model for context {context}: {len(y)} samples")

    def predict_property(self, smiles):
        """
        Predict molecular property with uncertainty estimate.

        Workflow:
        1. Extract context
        2. Featurize molecule
        3. If context has trained model:
           - Predict with RandomForest
           - Uncertainty = std dev across tree predictions
        4. Else (no model for context):
           - Return prior (ESOL mean ≈ -2.0 logS)
           - High uncertainty (2.0 logS)

        Args:
            smiles: SMILES string

        Returns:
            Tuple: (prediction, uncertainty, context)
            - prediction: float (predicted logS)
            - uncertainty: float (std dev estimate)
            - context: tuple (which context was used)
        """
        # Extract context
        context = self.get_context(smiles)

        # Handle invalid contexts
        if context[0] in ('invalid', 'unknown'):
            warnings.warn(f"Invalid context {context} for {smiles}")
            return -2.0, 999.0, context

        # Featurize
        try:
            features = self.featurize(smiles)
        except ValueError as e:
            warnings.warn(f"Could not featurize {smiles}: {e}")
            # Return prior with high uncertainty
            return -2.0, 999.0, context

        # Use context-specific model if available
        if context in self.context_models:
            model = self.context_models[context]

            # Prediction (reshape for sklearn: (1, n_features))
            prediction = model.predict([features])[0]

            # Uncertainty from tree variance
            tree_predictions = np.array([
                tree.predict([features])[0]
                for tree in model.estimators_
            ])
            uncertainty = np.std(tree_predictions)

            return prediction, uncertainty, context

        else:
            # No model for this context yet
            # Return ESOL prior: mean ≈ -2.0 logS, std ≈ 2.0 logS
            return -2.0, 2.0, context

    def get_statistics(self):
        """
        Get model statistics for all contexts.

        Returns:
            Dict with keys: 'contexts', 'counts', 'trained_models', 'total_samples'
        """
        stats = {
            'contexts': list(self.context_counts.keys()),
            'counts': self.context_counts,
            'trained_models': list(self.context_models.keys()),
            'total_samples': sum(self.context_counts.values())
        }
        return stats

    def save(self, filepath):
        """
        Save world model to disk (for persistence between episodes).

        Args:
            filepath: Path to save pickle file
        """
        model_data = {
            'belief_state': self.belief_state,
            'context_models': self.context_models,
            'context_counts': self.context_counts,
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'min_samples_context': self.min_samples_context
        }

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"Saved model to {filepath}")

    @classmethod
    def load(cls, filepath):
        """
        Load world model from disk.

        Args:
            filepath: Path to pickle file

        Returns:
            MolecularWorldModel instance
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        # Recreate model
        model = cls(
            n_estimators=model_data['n_estimators'],
            max_depth=model_data['max_depth'],
            min_samples_context=model_data['min_samples_context']
        )

        # Restore state
        model.belief_state = model_data['belief_state']
        model.context_models = model_data['context_models']
        model.context_counts = model_data['context_counts']

        print(f"Loaded model from {filepath}")
        return model


# ============================================================================
# TEST CODE
# ============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("MOLECULAR WORLD MODEL - UNIT TEST")
    print("=" * 70 + "\n")

    # Initialize model
    model = MolecularWorldModel()

    # Test molecules with known solubility (from ESOL dataset)
    test_data = [
        ('CCO', -0.77, 'Ethanol'),                          # aliphatic, small
        ('c1ccccc1', -0.87, 'Benzene'),                     # aromatic, small
        ('CC(C)(C)c1ccc(cc1)O', -5.17, 'BHT'),              # aromatic, large
        ('CCCCCCCCCCCCCCCCCCCCCCCCCCCCCC', -13.0, 'C30'),  # aliphatic, large
    ]

    print("1. Training on 4 molecules...")
    for smiles, logS, name in test_data:
        context = model.get_context(smiles)
        model.update_belief(smiles, logS)
        print(f"   {name:15s} ({smiles:30s})")
        print(f"      logS = {logS:6.2f}, context = {context}")

    print(f"\n2. Model statistics:")
    stats = model.get_statistics()
    print(f"   Total samples: {stats['total_samples']}")
    print(f"   Context counts: {stats['counts']}")
    print(f"   Trained models: {len(stats['trained_models'])} (need 50+ samples per context)")

    print(f"\n3. Predictions (with high uncertainty - only 1 sample per context):")
    for smiles, true_logS, name in test_data:
        pred, unc, ctx = model.predict_property(smiles)
        error = abs(pred - true_logS)
        print(f"   {name:15s}: pred = {pred:6.2f} ± {unc:5.2f}, true = {true_logS:6.2f}, error = {error:6.2f}")

    print(f"\n4. Test invalid SMILES:")
    try:
        pred, unc, ctx = model.predict_property("INVALID")
        print(f"   Invalid SMILES: pred = {pred:6.2f} ± {unc:5.2f}, context = {ctx}")
    except Exception as e:
        print(f"   Handled error: {e}")

    print(f"\n5. Test save/load functionality:")
    # Save model
    test_filepath = "/tmp/test_molecular_model.pkl"
    model.save(test_filepath)

    # Load model
    loaded_model = MolecularWorldModel.load(test_filepath)

    # Verify loaded model works
    pred_original, unc_original, ctx_original = model.predict_property('CCO')
    pred_loaded, unc_loaded, ctx_loaded = loaded_model.predict_property('CCO')

    print(f"   Original model prediction: {pred_original:6.2f} ± {unc_original:5.2f}")
    print(f"   Loaded model prediction:   {pred_loaded:6.2f} ± {unc_loaded:5.2f}")
    print(f"   Match: {abs(pred_original - pred_loaded) < 0.001}")

    # Clean up
    if os.path.exists(test_filepath):
        os.remove(test_filepath)
        print(f"   Cleaned up test file")

    print("\n" + "=" * 70)
    print("✅ MolecularWorldModel unit test complete!")
    print("=" * 70)
    print("\nNotes:")
    print("- High uncertainty is expected (only 1-4 samples per context)")
    print("- In Phase 2, train on 800 ESOL molecules → contexts have 126+ samples")
    print("- Then models will have low uncertainty (~0.1-0.3 logS)")
    print("\nNext steps:")
    print("1. Load full ESOL dataset (800 molecules)")
    print("2. Train model with update_belief() on training set")
    print("3. Evaluate with predict_property() on test set")
    print("4. Expected performance: MAE ~0.6-0.8 logS, R² ~0.8-0.85")
