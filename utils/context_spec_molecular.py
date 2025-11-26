"""
Molecular Context Specification for Drug Discovery

Defines 2-context scheme based on molecular aromaticity:
- Aromatic: Molecules with aromatic rings
- Aliphatic: Molecules without aromatic rings

This gives 2 contexts:
1. ('aromatic', 'all_sizes'): Has aromatic rings
2. ('aliphatic', 'all_sizes'): No aromatic rings

Validated on ESOL dataset (902 molecules):
- Aromatic: ~498 molecules (55.2%)
- Aliphatic: ~404 molecules (44.8%)
Both contexts have >400 samples → Very robust!
"""

from utils.context_spec import ContextSpec
from rdkit import Chem
from rdkit.Chem import Descriptors
from typing import Tuple, Dict
import warnings


def has_aromatic_rings(mol) -> bool:
    """
    Check if molecule has aromatic atoms.

    Args:
        mol: RDKit Mol object

    Returns:
        True if molecule contains aromatic atoms, False otherwise
    """
    if mol is None:
        return False

    for atom in mol.GetAtoms():
        if atom.GetIsAromatic():
            return True
    return False


def get_aromatic_type(mol) -> str:
    """
    Classify molecule as aromatic or aliphatic.

    Args:
        mol: RDKit Mol object

    Returns:
        'aromatic' if molecule has aromatic rings, 'aliphatic' otherwise
    """
    if mol is None:
        return 'invalid'

    return 'aromatic' if has_aromatic_rings(mol) else 'aliphatic'


def get_size_bin(mol, cutoff: float = 350.0) -> str:
    """
    Classify molecule size based on molecular weight.

    Args:
        mol: RDKit Mol object
        cutoff: Molecular weight cutoff in Daltons (default: 350)

    Returns:
        'small' if MW < cutoff, 'large' if MW >= cutoff
    """
    if mol is None:
        return 'invalid'

    try:
        mw = Descriptors.MolWt(mol)
        return 'small' if mw < cutoff else 'large'
    except Exception as e:
        warnings.warn(f"Error calculating molecular weight: {e}")
        return 'invalid'


def extract_molecular_context(observation: dict) -> Tuple[str, str]:
    """
    Extract molecular context from observation dictionary.

    Uses simplified 2-context scheme: aromatic vs. aliphatic only.
    Size information is ignored to ensure balanced contexts.

    Args:
        observation: Dictionary with 'smiles' key containing SMILES string

    Returns:
        Tuple of (aromatic_type, 'all_sizes'), e.g., ('aromatic', 'all_sizes')
        Returns ('invalid', 'invalid') if SMILES parsing fails
        Returns ('unknown', 'unknown') if 'smiles' key is missing
    """
    # Check if observation has SMILES
    if not isinstance(observation, dict):
        warnings.warn(f"Observation must be a dictionary, got {type(observation)}")
        return ('unknown', 'unknown')

    # Handle nested context structure (like toy domains)
    smiles = None
    if 'context' in observation and isinstance(observation['context'], dict):
        smiles = observation['context'].get('smiles')
    elif 'smiles' in observation:
        smiles = observation['smiles']

    if smiles is None:
        warnings.warn("No 'smiles' key found in observation")
        return ('unknown', 'unknown')

    # Parse SMILES with RDKit
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            warnings.warn(f"Invalid SMILES string: {smiles}")
            return ('invalid', 'invalid')

        # Extract aromatic type only (ignore size for simplified 2-context scheme)
        aromatic_type = get_aromatic_type(mol)

        # Return simplified context: (aromatic_type, 'all_sizes')
        return (aromatic_type, 'all_sizes')

    except Exception as e:
        warnings.warn(f"Error parsing SMILES '{smiles}': {e}")
        return ('invalid', 'invalid')


# Define the molecular context specification
MOLECULAR_CONTEXT = ContextSpec(
    name="molecular_properties",
    key_fn=extract_molecular_context
)


def validate_molecular_contexts(verbose: bool = True) -> bool:
    """
    Validate that ESOL dataset has balanced contexts.

    Loads ESOL dataset from DeepChem and checks that both 2 contexts
    (aromatic, aliphatic) have sufficient samples (>50 molecules recommended).

    Args:
        verbose: If True, print distribution statistics

    Returns:
        True if all contexts are well-represented, False if sparse contexts found
    """
    try:
        import deepchem as dc
        from collections import Counter
    except ImportError:
        warnings.warn("DeepChem not installed. Cannot validate ESOL contexts.")
        return False

    try:
        # Load ESOL dataset
        tasks, datasets, transformers = dc.molnet.load_delaney(featurizer='ECFP')
        train_dataset, valid_dataset, test_dataset = datasets

        # Combine all splits for full distribution
        all_smiles = []
        for dataset in [train_dataset, valid_dataset, test_dataset]:
            all_smiles.extend(dataset.ids)

        # Extract contexts for all molecules
        contexts = []
        invalid_count = 0
        for smiles in all_smiles:
            obs = {'smiles': smiles}
            context = extract_molecular_context(obs)
            if context[0] in ('invalid', 'unknown'):
                invalid_count += 1
            else:
                contexts.append(context)

        # Count distribution
        context_counts = Counter(contexts)
        total_valid = len(contexts)

        if verbose:
            print("=" * 60)
            print("ESOL Dataset Molecular Context Distribution")
            print("=" * 60)
            print(f"Total molecules: {len(all_smiles)}")
            print(f"Valid contexts: {total_valid}")
            print(f"Invalid/Unknown: {invalid_count}")
            print()
            print("Context Distribution:")
            print("-" * 60)

            # Sort by context for consistent display
            for context in sorted(context_counts.keys()):
                count = context_counts[context]
                percentage = (count / total_valid) * 100
                aromatic_type, size_bin = context
                print(f"  {aromatic_type:10s} × {size_bin:10s}: {count:4d} ({percentage:5.1f}%)")
            print("=" * 60)

        # Check for sparse contexts (< 50 samples as minimum threshold)
        sparse_threshold = 50
        sparse_contexts = [ctx for ctx, count in context_counts.items()
                          if count < sparse_threshold]

        if sparse_contexts:
            warnings.warn(
                f"Sparse contexts found (< {sparse_threshold} samples): {sparse_contexts}. "
                f"Consider using a different context scheme or collecting more data."
            )
            return False

        # Check that we have both expected contexts (2-context scheme)
        expected_contexts = {
            ('aromatic', 'all_sizes'),
            ('aliphatic', 'all_sizes')
        }

        found_contexts = set(context_counts.keys())
        missing_contexts = expected_contexts - found_contexts

        if missing_contexts:
            warnings.warn(f"Missing expected contexts: {missing_contexts}")
            return False

        if verbose:
            print(f"\n✓ Both contexts present with at least {sparse_threshold} samples each")
            print(f"✓ 2-context scheme validated for ESOL dataset\n")

        return True

    except Exception as e:
        warnings.warn(f"Error validating ESOL contexts: {e}")
        return False


if __name__ == "__main__":
    # Run validation when script is executed directly
    print("Validating molecular context specification on ESOL dataset...\n")
    is_valid = validate_molecular_contexts(verbose=True)

    if is_valid:
        print("✓ Validation passed: Context scheme is suitable for ESOL dataset")
    else:
        print("✗ Validation failed: Context scheme may need adjustment")

    # Test with sample SMILES
    print("\n" + "=" * 60)
    print("Testing with sample molecules:")
    print("=" * 60)

    test_molecules = [
        ("c1ccccc1", "Benzene (aromatic)"),
        ("CCCCCCCC", "Octane (aliphatic)"),
        ("c1ccc2c(c1)ccc3c2ccc4c3cccc4", "Tetracene (aromatic)"),
        ("CCCCCCCCCCCCCCCCCCCCCCCCCCCCCC", "Triacontane (aliphatic)"),
        ("invalid_smiles", "Invalid SMILES"),
    ]

    for smiles, description in test_molecules:
        obs = {'smiles': smiles}
        context = extract_molecular_context(obs)
        print(f"{description:30s} → {context}")

    print("=" * 60)
