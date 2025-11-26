"""
Quality check ESOL dataset and generate baseline statistics.

ESOL (Estimated SOLubility) is also known as the Delaney dataset.
It contains ~1100 molecules with measured aqueous solubility (logS units).
"""
from deepchem.molnet import load_delaney
from rdkit import Chem
from rdkit.Chem import Descriptors
import numpy as np
import pandas as pd
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.context_spec_molecular import extract_molecular_context


def main():
    print("=" * 60)
    print("ESOL Data Quality Check")
    print("=" * 60)

    # Load ESOL (Delaney dataset)
    tasks, datasets, transformers = load_delaney(featurizer='ECFP', split='scaffold')
    train_dataset, valid_dataset, test_dataset = datasets

    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_dataset)} molecules")
    print(f"  Valid: {len(valid_dataset)} molecules")
    print(f"  Test:  {len(test_dataset)} molecules")

    # Check parsing success
    def check_parsing(dataset, name):
        valid = []
        invalid = []
        for i in range(len(dataset)):
            smiles = dataset.ids[i]
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                invalid.append(smiles)
            else:
                try:
                    Chem.SanitizeMol(mol)
                    valid.append(smiles)
                except:
                    invalid.append(smiles)

        print(f"\n{name} parsing:")
        print(f"  Valid: {len(valid)} ({len(valid)/len(dataset)*100:.1f}%)")
        print(f"  Invalid: {len(invalid)} ({len(invalid)/len(dataset)*100:.1f}%)")
        if invalid:
            print(f"  First invalid: {invalid[0]}")
        return valid, invalid

    train_valid, train_invalid = check_parsing(train_dataset, "Train")
    test_valid, test_invalid = check_parsing(test_dataset, "Test")

    # Solubility distribution
    train_solubilities = [train_dataset.y[i][0] for i in range(len(train_dataset))]
    test_solubilities = [test_dataset.y[i][0] for i in range(len(test_dataset))]

    print(f"\nSolubility distribution (logS):")
    print(f"  Train: mean={np.mean(train_solubilities):.2f}, std={np.std(train_solubilities):.2f}")
    print(f"  Train: min={np.min(train_solubilities):.2f}, max={np.max(train_solubilities):.2f}")
    print(f"  Test:  mean={np.mean(test_solubilities):.2f}, std={np.std(test_solubilities):.2f}")

    # Context distribution
    train_contexts = []
    for smiles in train_valid:
        ctx = extract_molecular_context({'smiles': smiles})
        train_contexts.append(ctx)

    from collections import Counter
    context_counts = Counter(train_contexts)

    print(f"\nContext distribution (train):")
    for ctx, count in sorted(context_counts.items()):
        print(f"  {ctx}: {count} ({count/len(train_contexts)*100:.1f}%)")

    # Compute descriptor statistics for later use
    descriptors = {'MW': [], 'LogP': [], 'TPSA': []}

    for smiles in train_valid:
        mol = Chem.MolFromSmiles(smiles)
        descriptors['MW'].append(Descriptors.MolWt(mol))
        descriptors['LogP'].append(Descriptors.MolLogP(mol))
        descriptors['TPSA'].append(Descriptors.TPSA(mol))

    print(f"\nDescriptor statistics (train):")
    for prop in ['MW', 'LogP', 'TPSA']:
        values = descriptors[prop]
        print(f"  {prop}: mean={np.mean(values):.2f}, std={np.std(values):.2f}")
        print(f"    min={np.min(values):.2f}, max={np.max(values):.2f}")

    # Save descriptor stats for later use
    descriptor_stats = {
        prop: {
            'mean': float(np.mean(descriptors[prop])),
            'std': float(np.std(descriptors[prop])),
            'min': float(np.min(descriptors[prop])),
            'max': float(np.max(descriptors[prop]))
        }
        for prop in descriptors
    }

    import json
    os.makedirs('memory', exist_ok=True)
    with open('memory/esol_descriptor_stats.json', 'w') as f:
        json.dump(descriptor_stats, f, indent=2)

    print(f"\n✅ Saved descriptor stats to memory/esol_descriptor_stats.json")

    # Save cleaned datasets
    def save_dataset(dataset, valid_smiles, filename):
        data = []
        for i, smiles in enumerate(dataset.ids):
            if smiles in valid_smiles:
                data.append({
                    'smiles': smiles,
                    'solubility': float(dataset.y[i][0])
                })
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        print(f"✅ Saved {len(df)} molecules to {filename}")

    save_dataset(train_dataset, train_valid, 'memory/esol_train.csv')
    save_dataset(test_dataset, test_valid, 'memory/esol_test.csv')

    print("\n" + "=" * 60)
    print("Quality check complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
