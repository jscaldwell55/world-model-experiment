#!/usr/bin/env python3
"""
ESOL Data Processing Script

Downloads and processes the ESOL dataset for molecular world model experiments.
Creates candidate_pool (800 molecules) and test_set (300 molecules).

For each molecule computes:
- Morgan fingerprints (radius=2, nBits=1024)
- RDKit descriptors: MolWt, LogP, NumHDonors, NumHAcceptors, TPSA, NumRotatableBonds, NumAromaticRings
- Scaffold (Murcko scaffold)
"""

import os
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional

# RDKit imports
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import DataStructs

# DeepChem for data loading
import deepchem as dc


def download_esol_dataset() -> pd.DataFrame:
    """Download ESOL dataset using DeepChem."""
    print("Downloading ESOL dataset...")

    # Load ESOL from DeepChem
    tasks, datasets, transformers = dc.molnet.load_delaney(
        featurizer='Raw',
        splitter=None  # We'll do our own split
    )

    # Extract data
    dataset = datasets[0]
    smiles = dataset.ids
    y = dataset.y.flatten()

    df = pd.DataFrame({
        'smiles': smiles,
        'logS': y  # log solubility in mol/L
    })

    print(f"Downloaded {len(df)} molecules")
    return df


def compute_morgan_fingerprint(mol, radius: int = 2, n_bits: int = 1024) -> np.ndarray:
    """Compute Morgan fingerprint as numpy array."""
    if mol is None:
        return None

    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros(n_bits, dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def compute_descriptors(mol) -> Dict[str, float]:
    """Compute RDKit descriptors."""
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


def compute_murcko_scaffold(mol) -> str:
    """Compute Murcko scaffold SMILES."""
    if mol is None:
        return None

    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaffold)
    except:
        return None


def process_molecule(smiles: str) -> Optional[Dict]:
    """Process a single molecule, computing all features."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Compute features
    fp = compute_morgan_fingerprint(mol)
    descriptors = compute_descriptors(mol)
    scaffold = compute_murcko_scaffold(mol)

    if fp is None or descriptors is None:
        return None

    return {
        'smiles': smiles,
        'morgan_fp': fp,
        'descriptors': descriptors,
        'scaffold': scaffold
    }


def process_esol_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Process all molecules in the dataset."""
    print("Processing molecules...")

    processed_data = []
    failed = 0

    for idx, row in df.iterrows():
        result = process_molecule(row['smiles'])
        if result is not None:
            result['logS'] = row['logS']
            processed_data.append(result)
        else:
            failed += 1

    print(f"Processed {len(processed_data)} molecules, {failed} failed")

    # Convert to DataFrame with expanded descriptors
    records = []
    for item in processed_data:
        record = {
            'smiles': item['smiles'],
            'logS': item['logS'],
            'scaffold': item['scaffold'],
            'morgan_fp': item['morgan_fp']
        }
        record.update(item['descriptors'])
        records.append(record)

    return pd.DataFrame(records)


def create_train_test_split(df: pd.DataFrame,
                           candidate_size: int = 800,
                           random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into candidate_pool and test_set.

    Uses scaffold-aware splitting to ensure test set has some scaffold diversity.
    """
    print(f"Splitting data: {candidate_size} candidate, {len(df) - candidate_size} test")

    np.random.seed(random_state)

    # Group by scaffold
    scaffold_groups = df.groupby('scaffold').indices
    scaffolds = list(scaffold_groups.keys())
    np.random.shuffle(scaffolds)

    # Allocate scaffolds to test set until we have enough
    test_indices = []
    candidate_indices = []

    test_target = len(df) - candidate_size

    for scaffold in scaffolds:
        indices = list(scaffold_groups[scaffold])

        if len(test_indices) < test_target:
            # Prioritize smaller scaffold groups for test
            if len(indices) <= 5:
                test_indices.extend(indices)
            else:
                # Split large scaffold groups
                np.random.shuffle(indices)
                n_test = max(1, len(indices) // 4)
                test_indices.extend(indices[:n_test])
                candidate_indices.extend(indices[n_test:])
        else:
            candidate_indices.extend(indices)

    # If we have too many test indices, move some to candidate
    if len(test_indices) > test_target:
        excess = len(test_indices) - test_target
        np.random.shuffle(test_indices)
        candidate_indices.extend(test_indices[:excess])
        test_indices = test_indices[excess:]

    # If we have too few test indices, move some from candidate
    if len(test_indices) < test_target:
        needed = test_target - len(test_indices)
        np.random.shuffle(candidate_indices)
        test_indices.extend(candidate_indices[:needed])
        candidate_indices = candidate_indices[needed:]

    candidate_pool = df.iloc[candidate_indices].reset_index(drop=True)
    test_set = df.iloc[test_indices].reset_index(drop=True)

    print(f"Final split: {len(candidate_pool)} candidate, {len(test_set)} test")
    print(f"Unique scaffolds - candidate: {candidate_pool['scaffold'].nunique()}, test: {test_set['scaffold'].nunique()}")

    return candidate_pool, test_set


def compute_scaffold_fingerprints(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Compute Morgan fingerprints for scaffolds (for clustering)."""
    scaffold_fps = {}

    for scaffold in df['scaffold'].unique():
        if scaffold is None:
            continue
        mol = Chem.MolFromSmiles(scaffold)
        if mol is not None:
            fp = compute_morgan_fingerprint(mol)
            scaffold_fps[scaffold] = fp

    return scaffold_fps


def main():
    """Main processing pipeline."""
    # Create output directory
    os.makedirs('data', exist_ok=True)

    # Download dataset
    df_raw = download_esol_dataset()

    # Process molecules
    df_processed = process_esol_dataset(df_raw)

    # Create splits
    candidate_pool, test_set = create_train_test_split(df_processed)

    # Compute scaffold fingerprints for clustering
    all_data = pd.concat([candidate_pool, test_set], ignore_index=True)
    scaffold_fps = compute_scaffold_fingerprints(all_data)

    # Summary statistics
    print("\n=== Dataset Summary ===")
    print(f"Total molecules: {len(all_data)}")
    print(f"Candidate pool: {len(candidate_pool)}")
    print(f"Test set: {len(test_set)}")
    print(f"\nlogS statistics:")
    print(f"  Range: [{all_data['logS'].min():.2f}, {all_data['logS'].max():.2f}]")
    print(f"  Mean: {all_data['logS'].mean():.2f}")
    print(f"  Std: {all_data['logS'].std():.2f}")
    print(f"\nDescriptor ranges:")
    for desc in ['MolWt', 'LogP', 'TPSA', 'NumRotatableBonds']:
        print(f"  {desc}: [{all_data[desc].min():.1f}, {all_data[desc].max():.1f}]")
    print(f"\nUnique scaffolds: {all_data['scaffold'].nunique()}")

    # Save processed data
    output_data = {
        'candidate_pool': candidate_pool,
        'test_set': test_set,
        'scaffold_fps': scaffold_fps,
        'descriptor_columns': ['MolWt', 'LogP', 'NumHDonors', 'NumHAcceptors',
                               'TPSA', 'NumRotatableBonds', 'NumAromaticRings'],
        'metadata': {
            'n_candidate': len(candidate_pool),
            'n_test': len(test_set),
            'fp_radius': 2,
            'fp_nbits': 1024,
            'random_state': 42
        }
    }

    output_path = 'data/esol_processed.pkl'
    with open(output_path, 'wb') as f:
        pickle.dump(output_data, f)

    print(f"\nSaved processed data to {output_path}")

    return output_data


if __name__ == '__main__':
    main()
