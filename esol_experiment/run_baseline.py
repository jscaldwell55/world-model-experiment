"""
ESOL Experiment: Testing Offline Consolidation for Molecular Property Prediction

Two hypotheses to test:
1. CIRCULARITY: Self-training fails because pseudo-labels come from the same model
2. DIVERSITY: Self-training fails because synthetics don't cover the test distribution

This script sets up the baseline and data.
"""

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from rdkit.DataStructs import TanimotoSimilarity
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
import json
import os

# Paths
MEMORY_DIR = "/Users/jaycaldwell/world-model-experiment/esol_experiment/memory"
os.makedirs(MEMORY_DIR, exist_ok=True)

def get_esol_data():
    """
    Load ESOL dataset from RDKit's built-in data.
    ESOL = Estimated SOLubility (Delaney's aqueous solubility dataset)
    """
    from rdkit.Chem import PandasTools

    # ESOL is a classic benchmark - let's fetch it
    # Using the version from the DeepChem/MoleculeNet collection
    url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv"

    try:
        df = pd.read_csv(url)
        print(f"Loaded ESOL dataset: {len(df)} molecules")
        return df
    except Exception as e:
        print(f"Could not fetch from URL: {e}")
        # Fallback: generate synthetic ESOL-like data for testing
        return generate_fallback_esol()

def generate_fallback_esol():
    """Generate a small ESOL-like dataset if we can't fetch the real one."""
    smiles_list = [
        "CCO", "CCCO", "CCCCO", "CC(C)O", "CC(C)(C)O",
        "c1ccccc1", "c1ccccc1O", "c1ccccc1N", "c1ccccc1C",
        "CC(=O)O", "CC(=O)OC", "CCOC(=O)C", "c1ccc(O)cc1O",
        "CCN", "CCNC", "CCN(C)C", "c1ccncc1", "c1ccc2ncccc2c1",
        "CC(C)CC", "CCCCCC", "CCCCCCC", "CCCCCCCC",
        "c1ccc(Cl)cc1", "c1ccc(Br)cc1", "c1ccc(F)cc1",
        "CC(=O)Nc1ccccc1", "c1ccc(C(=O)O)cc1", "c1ccc(C(=O)N)cc1"
    ] * 35  # Expand to ~1000 molecules

    # Add some variations
    expanded = []
    for s in smiles_list:
        mol = Chem.MolFromSmiles(s)
        if mol:
            expanded.append(s)

    # Calculate "solubility" using the Delaney equation (what ESOL is based on)
    data = []
    for smi in expanded[:1000]:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            logp = Descriptors.MolLogP(mol)
            mw = Descriptors.MolWt(mol)
            rb = Descriptors.NumRotatableBonds(mol)
            ap = len(mol.GetAromaticAtoms()) / mol.GetNumAtoms() if mol.GetNumAtoms() > 0 else 0

            # Delaney equation approximation
            logs = 0.16 - 0.63*logp - 0.0062*mw + 0.066*rb - 0.74*ap
            logs += np.random.normal(0, 0.3)  # Add noise

            data.append({"smiles": smi, "measured log solubility in mols per litre": logs})

    return pd.DataFrame(data)

def compute_fingerprint(smiles, radius=2, n_bits=1024):
    """Compute Morgan fingerprint for a molecule."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)

def compute_descriptors(smiles):
    """Compute molecular descriptors for a molecule."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    return {
        'MolWt': Descriptors.MolWt(mol),
        'LogP': Descriptors.MolLogP(mol),
        'NumHDonors': Descriptors.NumHDonors(mol),
        'NumHAcceptors': Descriptors.NumHAcceptors(mol),
        'NumRotatableBonds': Descriptors.NumRotatableBonds(mol),
        'TPSA': Descriptors.TPSA(mol),
        'NumAromaticRings': Descriptors.NumAromaticRings(mol),
    }

def featurize(smiles):
    """Convert SMILES to feature vector (fingerprint + descriptors)."""
    fp = compute_fingerprint(smiles)
    desc = compute_descriptors(smiles)

    if fp is None or desc is None:
        return None

    # Convert fingerprint to array
    fp_arr = np.array(fp)

    # Convert descriptors to array
    desc_arr = np.array(list(desc.values()))

    return np.concatenate([fp_arr, desc_arr])

def compute_tanimoto_to_set(query_smiles, reference_smiles_list):
    """Compute max Tanimoto similarity of query to any molecule in reference set."""
    query_fp = compute_fingerprint(query_smiles)
    if query_fp is None:
        return 0.0

    max_sim = 0.0
    for ref_smi in reference_smiles_list:
        ref_fp = compute_fingerprint(ref_smi)
        if ref_fp is not None:
            sim = TanimotoSimilarity(query_fp, ref_fp)
            max_sim = max(max_sim, sim)

    return max_sim

def main():
    print("=" * 60)
    print("ESOL Offline Consolidation Experiment")
    print("=" * 60)

    # Load data
    print("\n1. Loading ESOL dataset...")
    df = get_esol_data()

    # Identify columns
    smiles_col = [c for c in df.columns if 'smiles' in c.lower()][0]
    target_col = [c for c in df.columns if 'solubility' in c.lower() or 'log' in c.lower()][0]

    print(f"   SMILES column: {smiles_col}")
    print(f"   Target column: {target_col}")

    # Clean data
    df = df[[smiles_col, target_col]].dropna()
    df.columns = ['smiles', 'solubility']

    # Validate SMILES
    valid_mask = df['smiles'].apply(lambda x: Chem.MolFromSmiles(x) is not None)
    df = df[valid_mask].reset_index(drop=True)
    print(f"   Valid molecules: {len(df)}")

    # Split data
    print("\n2. Splitting data (80/20)...")
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    print(f"   Training: {len(train_df)}, Test: {len(test_df)}")

    # Save splits
    train_df.to_csv(f"{MEMORY_DIR}/esol_train.csv", index=False)
    test_df.to_csv(f"{MEMORY_DIR}/esol_test.csv", index=False)

    # Featurize
    print("\n3. Featurizing molecules...")
    X_train = np.array([featurize(s) for s in train_df['smiles'] if featurize(s) is not None])
    y_train = train_df['solubility'].values[:len(X_train)]

    X_test = np.array([featurize(s) for s in test_df['smiles'] if featurize(s) is not None])
    y_test = test_df['solubility'].values[:len(X_test)]

    print(f"   X_train shape: {X_train.shape}")
    print(f"   X_test shape: {X_test.shape}")

    # Train baseline model
    print("\n4. Training baseline model (RandomForest)...")
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    train_metrics = {
        'mae': mean_absolute_error(y_train, y_pred_train),
        'rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
        'r2': r2_score(y_train, y_pred_train)
    }

    test_metrics = {
        'mae': mean_absolute_error(y_test, y_pred_test),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
        'r2': r2_score(y_test, y_pred_test)
    }

    print("\n5. Baseline Results:")
    print(f"   Training - MAE: {train_metrics['mae']:.3f}, RMSE: {train_metrics['rmse']:.3f}, R²: {train_metrics['r2']:.3f}")
    print(f"   Test     - MAE: {test_metrics['mae']:.3f}, RMSE: {test_metrics['rmse']:.3f}, R²: {test_metrics['r2']:.3f}")

    # Analyze distribution gap
    print("\n6. Analyzing train/test distribution gap...")
    train_smiles = train_df['smiles'].tolist()

    test_tanimotos = []
    for smi in test_df['smiles']:
        sim = compute_tanimoto_to_set(smi, train_smiles[:200])  # Sample for speed
        test_tanimotos.append(sim)

    test_tanimotos = np.array(test_tanimotos)

    print(f"   Test→Train Tanimoto: mean={np.mean(test_tanimotos):.3f}, min={np.min(test_tanimotos):.3f}, max={np.max(test_tanimotos):.3f}")
    print(f"   Extrapolative test points (Tanimoto < 0.4): {np.mean(test_tanimotos < 0.4)*100:.1f}%")

    # Save model and results
    print("\n7. Saving artifacts...")
    with open(f"{MEMORY_DIR}/baseline_model.pkl", 'wb') as f:
        pickle.dump(model, f)

    results = {
        'train_size': len(train_df),
        'test_size': len(test_df),
        'feature_dim': X_train.shape[1],
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'test_tanimoto_mean': float(np.mean(test_tanimotos)),
        'test_tanimoto_min': float(np.min(test_tanimotos)),
        'extrapolative_fraction': float(np.mean(test_tanimotos < 0.4))
    }

    with open(f"{MEMORY_DIR}/baseline_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n   Saved to {MEMORY_DIR}/")
    print("\nBaseline complete! Ready for augmentation experiments.")

    return results

if __name__ == "__main__":
    main()
