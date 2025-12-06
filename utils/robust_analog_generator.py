"""
Robust Analog Generation for ESOL using Direct RWMol Manipulation

Replaces SMARTS-based transforms with reliable molecular editing.
Generates analogs via direct atom/bond manipulation.
"""

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit import DataStructs
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
import random


class RobustAnalogGenerator:
    """
    Generate analogs via direct molecular editing (no SMARTS).

    Transforms that reliably work:
    1. Halogen addition (add F/Cl/Br to C with H)
    2. Methyl addition (add CH3 to C with H)
    3. Hydroxyl addition (add OH to C with H)
    4. Halogen swapping (F↔Cl↔Br)
    5. Methyl removal (remove terminal CH3)
    """

    def __init__(self, training_smiles: List[str]):
        self.training_smiles = training_smiles
        self.training_fps = self._precompute_fingerprints(training_smiles)

    def _precompute_fingerprints(self, smiles_list: List[str]):
        """Precompute Morgan fingerprints for fast similarity."""
        fps = []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                fps.append(fp)
        return fps

    def generate_analogs(self, smiles: str, n_analogs: int = 3) -> List[Tuple[str, str]]:
        """
        Generate analogs for a single molecule.

        Returns: List of (analog_smiles, transform_name)
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return []

        analogs = []
        transforms = [
            ('add_fluorine', self._add_halogen, {'halogen': 'F'}),
            ('add_chlorine', self._add_halogen, {'halogen': 'Cl'}),
            ('add_bromine', self._add_halogen, {'halogen': 'Br'}),
            ('add_methyl', self._add_methyl, {}),
            ('add_hydroxyl', self._add_hydroxyl, {}),
            ('swap_halogen', self._swap_halogen, {}),
            ('remove_methyl', self._remove_methyl, {}),
        ]

        random.shuffle(transforms)

        for name, func, kwargs in transforms:
            if len(analogs) >= n_analogs:
                break
            try:
                result = func(mol, **kwargs)
                if result is not None:
                    analog_smi = Chem.MolToSmiles(result)
                    # Validate: can we parse it back?
                    if Chem.MolFromSmiles(analog_smi) is not None:
                        if analog_smi != smiles:  # Not identical to parent
                            analogs.append((analog_smi, name))
            except Exception:
                continue

        return analogs

    def _get_carbons_with_h(self, mol) -> List[int]:
        """Find carbon atoms that have at least one hydrogen."""
        carbons = []
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 6:  # Carbon
                if atom.GetTotalNumHs() > 0:
                    carbons.append(atom.GetIdx())
        return carbons

    def _add_halogen(self, mol, halogen: str = 'Cl') -> Optional[Chem.Mol]:
        """Add halogen to a carbon with available H."""
        halogen_num = {'F': 9, 'Cl': 17, 'Br': 35, 'I': 53}[halogen]

        carbons = self._get_carbons_with_h(mol)
        if not carbons:
            return None

        target_idx = random.choice(carbons)

        rwmol = Chem.RWMol(mol)
        new_atom_idx = rwmol.AddAtom(Chem.Atom(halogen_num))
        rwmol.AddBond(target_idx, new_atom_idx, Chem.BondType.SINGLE)

        try:
            Chem.SanitizeMol(rwmol)
            return rwmol.GetMol()
        except:
            return None

    def _add_methyl(self, mol) -> Optional[Chem.Mol]:
        """Add -CH3 to a carbon with available H."""
        carbons = self._get_carbons_with_h(mol)
        if not carbons:
            return None

        target_idx = random.choice(carbons)

        rwmol = Chem.RWMol(mol)
        new_c_idx = rwmol.AddAtom(Chem.Atom(6))  # Carbon
        rwmol.AddBond(target_idx, new_c_idx, Chem.BondType.SINGLE)

        try:
            Chem.SanitizeMol(rwmol)
            return rwmol.GetMol()
        except:
            return None

    def _add_hydroxyl(self, mol) -> Optional[Chem.Mol]:
        """Add -OH to a carbon with available H."""
        carbons = self._get_carbons_with_h(mol)
        if not carbons:
            return None

        target_idx = random.choice(carbons)

        rwmol = Chem.RWMol(mol)
        new_o_idx = rwmol.AddAtom(Chem.Atom(8))  # Oxygen
        rwmol.AddBond(target_idx, new_o_idx, Chem.BondType.SINGLE)

        try:
            Chem.SanitizeMol(rwmol)
            return rwmol.GetMol()
        except:
            return None

    def _swap_halogen(self, mol) -> Optional[Chem.Mol]:
        """Swap one halogen for another."""
        halogen_map = {9: [17, 35], 17: [9, 35], 35: [9, 17]}  # F, Cl, Br

        halogens = []
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() in halogen_map:
                halogens.append(atom.GetIdx())

        if not halogens:
            return None

        target_idx = random.choice(halogens)

        rwmol = Chem.RWMol(mol)
        old_num = rwmol.GetAtomWithIdx(target_idx).GetAtomicNum()
        new_num = random.choice(halogen_map[old_num])
        rwmol.GetAtomWithIdx(target_idx).SetAtomicNum(new_num)

        try:
            Chem.SanitizeMol(rwmol)
            return rwmol.GetMol()
        except:
            return None

    def _remove_methyl(self, mol) -> Optional[Chem.Mol]:
        """Remove a terminal -CH3 group."""
        # Find terminal carbons (degree 1, only C-C bond)
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 6 and atom.GetDegree() == 1:
                neighbors = atom.GetNeighbors()
                if len(neighbors) > 0:
                    neighbor = neighbors[0]
                    if neighbor.GetAtomicNum() == 6:  # C-C bond
                        rwmol = Chem.RWMol(mol)
                        rwmol.RemoveAtom(atom.GetIdx())
                        try:
                            Chem.SanitizeMol(rwmol)
                            return rwmol.GetMol()
                        except:
                            return None
        return None

    def compute_diversity_score(self, smiles: str) -> float:
        """
        Diversity = 1 - max_similarity_to_training
        Higher = more novel
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 0.0

        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)

        max_sim = 0.0
        for train_fp in self.training_fps:
            sim = DataStructs.TanimotoSimilarity(fp, train_fp)
            max_sim = max(max_sim, sim)

        return 1.0 - max_sim

    def compute_properties(self, smiles: str) -> dict:
        """Compute MW, LogP for bias targeting."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {'MW': 0, 'LogP': 0, 'valid': False}

        return {
            'MW': Descriptors.MolWt(mol),
            'LogP': Descriptors.MolLogP(mol),
            'valid': True
        }


def generate_diverse_analogs(
    training_df: pd.DataFrame,
    n_candidates: int = 1200,
    n_select: int = 600,
    bias_mw_threshold: float = 337.46,
    bias_logp_threshold: float = 4.20,
    min_diversity: float = 0.3,  # Keep if diversity > 0.3 (sim < 0.7)
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Main function: Generate and select diverse analogs.

    Strategy:
    1. Generate 3 analogs per parent (covering ~1200 candidates from 400 parents)
    2. Score by diversity (distance from training)
    3. Bonus for landing in bias regions
    4. Greedy selection to maximize coverage

    Returns:
        (candidates_df, selected_df)
    """

    print("=" * 80)
    print("ROBUST ANALOG GENERATION")
    print("=" * 80)

    training_smiles = training_df['smiles'].tolist()
    generator = RobustAnalogGenerator(training_smiles)

    print(f"\nInitialized generator with {len(training_smiles)} training molecules")

    # Prioritize parents near bias thresholds
    print("\nPrioritizing parents near bias thresholds...")
    parent_priority = []
    for _, row in training_df.iterrows():
        smi = row['smiles']
        props = generator.compute_properties(smi)

        if not props['valid']:
            continue

        # Score: Higher if close to (but below) bias thresholds
        mw_dist = abs(props['MW'] - bias_mw_threshold)
        logp_dist = abs(props['LogP'] - bias_logp_threshold)

        mw_score = max(0, 1 - mw_dist / 100)
        logp_score = max(0, 1 - abs(logp_dist) / 2)
        priority = mw_score + logp_score

        parent_priority.append((smi, priority, props['MW'], props['LogP']))

    # Sort by priority (high first)
    parent_priority.sort(key=lambda x: -x[1])

    print(f"  Top priority parent: MW={parent_priority[0][2]:.1f}, LogP={parent_priority[0][3]:.2f}, priority={parent_priority[0][1]:.3f}")

    # Generate candidates
    print(f"\nGenerating {n_candidates} candidate analogs...")
    candidates = []
    seen_smiles = set(training_smiles)

    n_parents_to_use = min(len(parent_priority), n_candidates // 3 + 100)

    for i, (parent_smi, _, _, _) in enumerate(parent_priority[:n_parents_to_use]):
        analogs = generator.generate_analogs(parent_smi, n_analogs=4)  # Generate 4, take best 3

        for analog_smi, transform in analogs:
            if analog_smi in seen_smiles:
                continue
            seen_smiles.add(analog_smi)

            props = generator.compute_properties(analog_smi)
            if not props['valid']:
                continue

            diversity = generator.compute_diversity_score(analog_smi)

            # Skip if too similar to training
            if diversity < min_diversity:
                continue

            in_mw_bias = props['MW'] > bias_mw_threshold
            in_logp_bias = props['LogP'] > bias_logp_threshold

            # Combined score: diversity + bias bonus
            bias_bonus = 0.3 if (in_mw_bias or in_logp_bias) else 0.0
            score = diversity + bias_bonus

            candidates.append({
                'smiles': analog_smi,
                'parent_smiles': parent_smi,
                'transform': transform,
                'MW': props['MW'],
                'LogP': props['LogP'],
                'diversity': diversity,
                'in_mw_bias': in_mw_bias,
                'in_logp_bias': in_logp_bias,
                'score': score
            })

        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{n_parents_to_use} parents, {len(candidates)} candidates so far...")

    candidates_df = pd.DataFrame(candidates)

    print(f"\n✅ Generated {len(candidates_df)} valid candidates")
    if len(candidates_df) > 0:
        print(f"   Diversity: mean={candidates_df['diversity'].mean():.3f}, median={candidates_df['diversity'].median():.3f}")
        print(f"   In MW bias: {candidates_df['in_mw_bias'].sum()} ({candidates_df['in_mw_bias'].mean()*100:.1f}%)")
        print(f"   In LogP bias: {candidates_df['in_logp_bias'].sum()} ({candidates_df['in_logp_bias'].mean()*100:.1f}%)")

    if len(candidates_df) == 0:
        print("\n⚠️  No valid candidates generated!")
        return candidates_df, pd.DataFrame()

    # Greedy selection for diversity
    print(f"\nSelecting {n_select} most diverse analogs...")
    selected = greedy_diverse_selection(
        candidates_df,
        n_select=n_select,
        training_smiles=training_smiles,
        bias_target_fraction=0.25
    )

    return candidates_df, selected


def greedy_diverse_selection(
    candidates_df: pd.DataFrame,
    n_select: int,
    training_smiles: List[str],
    bias_target_fraction: float = 0.25
) -> pd.DataFrame:
    """
    Greedy selection that:
    1. Maximizes diversity from training AND from already-selected
    2. Ensures ~25% are in bias regions
    """

    # Precompute candidate fingerprints
    print("  Computing fingerprints for selection...")
    candidate_fps = {}
    for _, row in candidates_df.iterrows():
        mol = Chem.MolFromSmiles(row['smiles'])
        if mol:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            candidate_fps[row['smiles']] = fp

    selected_indices = []
    selected_fps = []

    # Reserve slots for bias region molecules
    n_bias_target = int(n_select * bias_target_fraction)

    print(f"  Target: {n_bias_target} in bias regions, {n_select - n_bias_target} diverse")

    # First pass: select bias region molecules
    print("  Phase 1: Selecting bias region molecules...")
    bias_candidates = candidates_df[
        candidates_df['in_mw_bias'] | candidates_df['in_logp_bias']
    ].sort_values('score', ascending=False)

    n_bias_selected = 0
    for idx, row in bias_candidates.iterrows():
        if n_bias_selected >= n_bias_target:
            break

        smi = row['smiles']
        fp = candidate_fps.get(smi)
        if fp is None:
            continue

        # Check not too similar to already selected
        too_similar = False
        for sel_fp in selected_fps:
            if DataStructs.TanimotoSimilarity(fp, sel_fp) > 0.8:
                too_similar = True
                break

        if not too_similar:
            selected_indices.append(idx)
            selected_fps.append(fp)
            n_bias_selected += 1

    print(f"    Selected {n_bias_selected} bias region molecules")

    # Second pass: fill remaining with most diverse
    print("  Phase 2: Filling with diverse molecules...")
    remaining = candidates_df[~candidates_df.index.isin(selected_indices)]
    remaining = remaining.sort_values('diversity', ascending=False)

    for idx, row in remaining.iterrows():
        if len(selected_indices) >= n_select:
            break

        smi = row['smiles']
        fp = candidate_fps.get(smi)
        if fp is None:
            continue

        # Check not too similar to already selected
        too_similar = False
        for sel_fp in selected_fps:
            if DataStructs.TanimotoSimilarity(fp, sel_fp) > 0.8:
                too_similar = True
                break

        if not too_similar:
            selected_indices.append(idx)
            selected_fps.append(fp)

        if len(selected_indices) % 100 == 0:
            print(f"    Selected {len(selected_indices)}/{n_select}...")

    selected_df = candidates_df.loc[selected_indices].copy()

    print(f"\n✅ Selected {len(selected_df)} diverse analogs")
    print(f"   In MW bias region: {selected_df['in_mw_bias'].sum()} ({selected_df['in_mw_bias'].mean()*100:.1f}%)")
    print(f"   In LogP bias region: {selected_df['in_logp_bias'].sum()} ({selected_df['in_logp_bias'].mean()*100:.1f}%)")
    print(f"   Mean diversity: {selected_df['diversity'].mean():.3f}")
    print(f"   Median diversity: {selected_df['diversity'].median():.3f}")

    return selected_df


if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # Test with ESOL data
    train_df = pd.read_csv('memory/esol_train.csv')

    print("Testing robust analog generator...")
    candidates_df, result_df = generate_diverse_analogs(
        train_df,
        n_candidates=1200,
        n_select=600
    )

    if len(candidates_df) > 0:
        candidates_df.to_csv('memory/esol_synthetics_candidates_robust.csv', index=False)
        print(f"\n✅ Saved {len(candidates_df)} candidates to memory/esol_synthetics_candidates_robust.csv")

    if len(result_df) > 0:
        result_df.to_csv('memory/esol_synthetics_filtered_robust.csv', index=False)
        print(f"✅ Saved {len(result_df)} selected analogs to memory/esol_synthetics_filtered_robust.csv")
