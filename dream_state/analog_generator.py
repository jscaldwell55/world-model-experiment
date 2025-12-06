"""
AnalogGenerator: Generate virtual molecular analogs via RDKit transformations.

Implements robust molecular transformations to explore chemical space:
1. Halogen Swap: Mutate Cl <-> F <-> Br at aromatic positions
2. Methylation: Add -CH3 to aromatic carbons with available H
3. Hydroxyl Addition: Add -OH to aromatic positions

All transformations include validation via Chem.SanitizeMol() to ensure
chemically valid outputs.
"""

import logging
from typing import Dict, List, Optional, Set
import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit import RDLogger

# Suppress RDKit warnings
RDLogger.DisableLog('rdApp.*')

logger = logging.getLogger(__name__)


class AnalogGenerator:
    """
    Generate virtual molecular analogs via chemical transformations.

    Transformations:
    - Halogen swap: F <-> Cl <-> Br
    - Methylation: Add -CH3 to aromatic carbons
    - Hydroxyl addition: Add -OH to aromatic carbons

    All outputs are validated for chemical validity and deduplicated.
    """

    # Halogen atoms for swapping
    HALOGENS = {'F': 9, 'Cl': 17, 'Br': 35}
    HALOGEN_SWAPS = {
        'F': ['Cl', 'Br'],
        'Cl': ['F', 'Br'],
        'Br': ['F', 'Cl']
    }

    def __init__(self, random_state: int = 42):
        """
        Initialize the analog generator.

        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)

    def generate_analogs(
        self,
        smiles: str,
        n_variants: int = 5
    ) -> List[Dict]:
        """
        Generate molecular analogs for a given SMILES.

        Args:
            smiles: Parent SMILES string
            n_variants: Maximum number of variants to return

        Returns:
            List of dicts with keys:
            - smiles: Analog SMILES
            - parent_smiles: Original SMILES
            - transformation: Description of transformation applied
        """
        # Parse parent molecule
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.debug(f"Invalid parent SMILES: {smiles}")
            return []

        # Collect all possible analogs
        all_analogs = []
        seen_smiles: Set[str] = {smiles}  # Track to avoid duplicates

        # Apply transformations
        halogen_analogs = self._halogen_swap(mol, smiles)
        methyl_analogs = self._methylation(mol, smiles)
        hydroxyl_analogs = self._hydroxyl_addition(mol, smiles)

        # Combine and deduplicate
        for analog in halogen_analogs + methyl_analogs + hydroxyl_analogs:
            analog_smiles = analog['smiles']
            if analog_smiles not in seen_smiles:
                seen_smiles.add(analog_smiles)
                all_analogs.append(analog)

        # Shuffle and limit
        if len(all_analogs) > n_variants:
            self.rng.shuffle(all_analogs)
            all_analogs = all_analogs[:n_variants]

        return all_analogs

    def _halogen_swap(self, mol: Chem.Mol, parent_smiles: str) -> List[Dict]:
        """
        Swap halogens (F <-> Cl <-> Br) at all positions.

        Returns list of valid analog dicts.
        """
        analogs = []

        for atom in mol.GetAtoms():
            symbol = atom.GetSymbol()
            if symbol in self.HALOGEN_SWAPS:
                atom_idx = atom.GetIdx()

                for new_halogen in self.HALOGEN_SWAPS[symbol]:
                    # Create editable copy
                    new_mol = Chem.RWMol(mol)

                    # Replace atom
                    new_mol.GetAtomWithIdx(atom_idx).SetAtomicNum(
                        self.HALOGENS[new_halogen]
                    )

                    # Validate
                    analog_smiles = self._validate_and_canonicalize(new_mol)
                    if analog_smiles and analog_smiles != parent_smiles:
                        analogs.append({
                            'smiles': analog_smiles,
                            'parent_smiles': parent_smiles,
                            'transformation': f'halogen_swap_{symbol}_to_{new_halogen}'
                        })

        return analogs

    def _methylation(self, mol: Chem.Mol, parent_smiles: str) -> List[Dict]:
        """
        Add -CH3 to aromatic carbons with available hydrogens.

        Returns list of valid analog dicts.
        """
        analogs = []

        for atom in mol.GetAtoms():
            # Find aromatic carbons with at least one hydrogen
            if (atom.GetSymbol() == 'C' and
                atom.GetIsAromatic() and
                atom.GetTotalNumHs() > 0):

                atom_idx = atom.GetIdx()

                # Create editable copy
                new_mol = Chem.RWMol(mol)

                # Add methyl carbon
                methyl_idx = new_mol.AddAtom(Chem.Atom(6))  # Carbon
                new_mol.AddBond(atom_idx, methyl_idx, Chem.BondType.SINGLE)

                # Validate
                analog_smiles = self._validate_and_canonicalize(new_mol)
                if analog_smiles and analog_smiles != parent_smiles:
                    analogs.append({
                        'smiles': analog_smiles,
                        'parent_smiles': parent_smiles,
                        'transformation': f'methylation_at_C{atom_idx}'
                    })

        return analogs

    def _hydroxyl_addition(self, mol: Chem.Mol, parent_smiles: str) -> List[Dict]:
        """
        Add -OH to aromatic carbons with available hydrogens.

        Returns list of valid analog dicts.
        """
        analogs = []

        for atom in mol.GetAtoms():
            # Find aromatic carbons with at least one hydrogen
            if (atom.GetSymbol() == 'C' and
                atom.GetIsAromatic() and
                atom.GetTotalNumHs() > 0):

                atom_idx = atom.GetIdx()

                # Create editable copy
                new_mol = Chem.RWMol(mol)

                # Add oxygen
                oxygen_idx = new_mol.AddAtom(Chem.Atom(8))  # Oxygen
                new_mol.AddBond(atom_idx, oxygen_idx, Chem.BondType.SINGLE)

                # Validate
                analog_smiles = self._validate_and_canonicalize(new_mol)
                if analog_smiles and analog_smiles != parent_smiles:
                    analogs.append({
                        'smiles': analog_smiles,
                        'parent_smiles': parent_smiles,
                        'transformation': f'hydroxyl_addition_at_C{atom_idx}'
                    })

        return analogs

    def _validate_and_canonicalize(self, mol: Chem.RWMol) -> Optional[str]:
        """
        Validate molecule and return canonical SMILES if valid.

        Returns None if molecule is invalid.
        """
        try:
            # Convert to regular Mol
            mol = mol.GetMol()

            # Sanitize (validates chemistry)
            Chem.SanitizeMol(mol)

            # Return canonical SMILES
            return Chem.MolToSmiles(mol)
        except Exception:
            return None

    def get_transformation_stats(
        self,
        smiles_list: List[str],
        n_variants: int = 5
    ) -> Dict:
        """
        Get statistics on transformations across a set of molecules.

        Useful for understanding the diversity of generated analogs.
        """
        stats = {
            'n_molecules': len(smiles_list),
            'n_valid_parents': 0,
            'n_analogs_total': 0,
            'n_halogen_swaps': 0,
            'n_methylations': 0,
            'n_hydroxyl_additions': 0,
            'avg_analogs_per_molecule': 0.0
        }

        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue

            stats['n_valid_parents'] += 1

            analogs = self.generate_analogs(smiles, n_variants)
            stats['n_analogs_total'] += len(analogs)

            for analog in analogs:
                transform = analog['transformation']
                if 'halogen_swap' in transform:
                    stats['n_halogen_swaps'] += 1
                elif 'methylation' in transform:
                    stats['n_methylations'] += 1
                elif 'hydroxyl' in transform:
                    stats['n_hydroxyl_additions'] += 1

        if stats['n_valid_parents'] > 0:
            stats['avg_analogs_per_molecule'] = (
                stats['n_analogs_total'] / stats['n_valid_parents']
            )

        return stats
