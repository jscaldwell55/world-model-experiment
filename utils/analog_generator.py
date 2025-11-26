"""
Simple molecular analog generator for synthetic data creation.

Generates analogs by applying simple SMILES transformations.
"""

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
import random
from typing import List, Tuple


def generate_molecular_analogs_v0(parent_smiles: str,
                                  n_analogs: int = 5,
                                  tanimoto_range: Tuple[float, float] = (0.6, 0.85),
                                  max_attempts: int = 100) -> List[Tuple[str, str, str]]:
    """
    Generate molecular analogs through simple transformations.

    V0 strategy: Random substituent changes, ring modifications, etc.

    Args:
        parent_smiles: Parent molecule SMILES
        n_analogs: Number of analogs to generate
        tanimoto_range: Desired Tanimoto similarity range to parent
        max_attempts: Maximum attempts per analog

    Returns:
        List of (analog_smiles, transform_type, reason) tuples
    """
    parent_mol = Chem.MolFromSmiles(parent_smiles)
    if parent_mol is None:
        return []

    parent_fp = AllChem.GetMorganFingerprintAsBitVect(parent_mol, 2, nBits=2048)

    # Simple transformation strategies
    transformations = [
        ('methyl_add', lambda mol: add_methyl_group(mol)),
        ('methyl_remove', lambda mol: remove_methyl_group(mol)),
        ('halogen_swap', lambda mol: swap_halogen(mol)),
        ('ring_extend', lambda mol: extend_ring(mol)),
    ]

    analogs = []
    attempts = 0

    while len(analogs) < n_analogs and attempts < max_attempts * n_analogs:
        attempts += 1

        # Pick random transformation
        transform_name, transform_fn = random.choice(transformations)

        try:
            analog_mol = transform_fn(parent_mol)
            if analog_mol is None:
                continue

            # Check validity
            analog_smiles = Chem.MolToSmiles(analog_mol)
            if analog_smiles == parent_smiles:
                continue

            # Check Tanimoto similarity
            analog_fp = AllChem.GetMorganFingerprintAsBitVect(analog_mol, 2, nBits=2048)
            similarity = DataStructs.TanimotoSimilarity(parent_fp, analog_fp)

            if tanimoto_range[0] <= similarity <= tanimoto_range[1]:
                reason = f"Applied {transform_name}, similarity={similarity:.2f}"
                analogs.append((analog_smiles, transform_name, reason))

        except Exception:
            continue

    return analogs


def add_methyl_group(mol):
    """Add a methyl group to a random carbon."""
    mol_copy = Chem.RWMol(mol)

    # Find carbons that can accept a methyl
    carbon_indices = [atom.GetIdx() for atom in mol_copy.GetAtoms()
                     if atom.GetSymbol() == 'C' and atom.GetTotalDegree() < 4]

    if not carbon_indices:
        return None

    # Pick random carbon
    carbon_idx = random.choice(carbon_indices)

    # Add methyl group
    methyl_idx = mol_copy.AddAtom(Chem.Atom(6))  # Carbon
    mol_copy.AddBond(carbon_idx, methyl_idx, Chem.BondType.SINGLE)

    try:
        Chem.SanitizeMol(mol_copy)
        return mol_copy.GetMol()
    except:
        return None


def remove_methyl_group(mol):
    """Remove a methyl group from a random position."""
    mol_copy = Chem.RWMol(mol)

    # Find terminal methyl groups (C with 1 neighbor, all H)
    methyl_indices = []
    for atom in mol_copy.GetAtoms():
        if atom.GetSymbol() == 'C' and atom.GetDegree() == 1:
            neighbor = atom.GetNeighbors()[0]
            if neighbor.GetSymbol() == 'C':
                methyl_indices.append(atom.GetIdx())

    if not methyl_indices:
        return None

    # Remove random methyl
    methyl_idx = random.choice(methyl_indices)
    mol_copy.RemoveAtom(methyl_idx)

    try:
        Chem.SanitizeMol(mol_copy)
        return mol_copy.GetMol()
    except:
        return None


def swap_halogen(mol):
    """Swap a halogen for another halogen."""
    mol_copy = Chem.RWMol(mol)

    halogens = ['F', 'Cl', 'Br', 'I']

    # Find existing halogens
    halogen_indices = [atom.GetIdx() for atom in mol_copy.GetAtoms()
                      if atom.GetSymbol() in halogens]

    if not halogen_indices:
        return None

    # Pick random halogen to swap
    halogen_idx = random.choice(halogen_indices)
    current_halogen = mol_copy.GetAtomWithIdx(halogen_idx).GetSymbol()

    # Pick different halogen
    new_halogens = [h for h in halogens if h != current_halogen]
    new_halogen = random.choice(new_halogens)

    # Replace
    mol_copy.GetAtomWithIdx(halogen_idx).SetAtomicNum(
        {'F': 9, 'Cl': 17, 'Br': 35, 'I': 53}[new_halogen]
    )

    try:
        Chem.SanitizeMol(mol_copy)
        return mol_copy.GetMol()
    except:
        return None


def extend_ring(mol):
    """Add a simple ring extension (e.g., fuse benzene)."""
    # Simple implementation: add a benzene ring to an aromatic carbon
    mol_copy = Chem.RWMol(mol)

    # Find aromatic carbons
    aromatic_carbons = [atom.GetIdx() for atom in mol_copy.GetAtoms()
                       if atom.GetIsAromatic() and atom.GetSymbol() == 'C']

    if not aromatic_carbons:
        return None

    # This is complex to do properly, so for v0 just return None
    # In a real implementation, would use RDKit reaction templates
    return None


if __name__ == "__main__":
    # Test analog generation
    test_smiles = [
        'c1ccccc1',  # Benzene
        'CCO',       # Ethanol
        'CC(C)Cl',   # Isopropyl chloride
    ]

    print("Testing analog generator...")
    for smiles in test_smiles:
        print(f"\nParent: {smiles}")
        analogs = generate_molecular_analogs_v0(smiles, n_analogs=3)
        for analog_smiles, transform, reason in analogs:
            print(f"  Analog: {analog_smiles}")
            print(f"    {reason}")
