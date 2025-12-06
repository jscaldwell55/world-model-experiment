"""
Diversity-Maximizing Analog Generator for ESOL

Generates synthetic molecules that FILL GAPS in chemical space rather than
densifying the training distribution.

Strategy:
1. Generate candidate analogs from training molecules
2. Score by diversity (distance to training set)
3. Oversample in bias regions (MW>337, LogP>4.2)
4. Select diverse subset that maximizes coverage

Key insight from diagnostics:
- 60% of test molecules have Tanimoto < 0.4 to training (extrapolation)
- Test oversamples MW>337 (2.7x) and LogP>4.2 (2.0x)
- Oracle's advantage is from COVERAGE, not novelty
"""

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, DataStructs
from typing import List, Dict, Tuple, Optional
import random
from collections import defaultdict


# Transformation library for analog generation
# Each transform is (SMARTS_pattern, SMARTS_replacement, name, expected_MW_delta, expected_LogP_delta)
TRANSFORMS = [
    # Add functional groups
    ("[CH3]", "[CH2]C", "extend_alkyl", 14, 0.5),
    ("[CH2][CH3]", "[CH2][CH2][CH3]", "extend_chain", 14, 0.5),
    ("c", "c(Cl)", "add_chloro_aromatic", 35, 0.5),
    ("[CH]", "[C](F)", "add_fluoro", 18, 0.14),
    ("c1ccccc1", "c1ccc(C)cc1", "add_methyl_aromatic", 14, 0.5),

    # Add rings (pushes MW up significantly)
    ("[CH2]", "[CH2]c1ccccc1", "add_phenyl", 77, 1.5),
    ("C", "C1CCCCC1", "cyclize_to_cyclohexyl", 56, 1.2),

    # Add heteroatoms
    ("CC", "CCO", "add_hydroxyl", 16, -1.0),
    ("CC", "CCN", "add_amino", 15, -1.3),
    ("C(=O)O", "C(=O)OC", "esterify", 14, 0.6),

    # Larger modifications for bias region targeting
    ("[H]", "C(C)(C)C", "add_tert_butyl", 57, 1.7),
    ("c", "c(C(F)(F)F)", "add_CF3", 69, 1.1),
]


def get_morgan_fp(smiles: str):
    """Get Morgan fingerprint for similarity calculations."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)


def compute_max_similarity_to_set(smiles: str, reference_fps: List) -> float:
    """Compute maximum Tanimoto similarity to a set of reference fingerprints."""
    fp = get_morgan_fp(smiles)
    if fp is None:
        return 1.0  # Invalid SMILES, mark as highly similar (will be filtered)

    if len(reference_fps) == 0:
        return 0.0

    similarities = [DataStructs.TanimotoSimilarity(fp, ref_fp) for ref_fp in reference_fps]
    return max(similarities)


def get_molecular_descriptors(smiles: str) -> Optional[Dict]:
    """Get molecular descriptors."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    return {
        'MW': Descriptors.MolWt(mol),
        'LogP': Descriptors.MolLogP(mol),
        'TPSA': Descriptors.TPSA(mol),
        'NumHDonors': Descriptors.NumHDonors(mol),
        'NumHAcceptors': Descriptors.NumHAcceptors(mol),
        'NumRotatableBonds': Descriptors.NumRotatableBonds(mol),
        'NumAromaticRings': Descriptors.NumAromaticRings(mol),
        'NumAliphaticRings': Descriptors.NumAliphaticRings(mol)
    }


def apply_transform(smiles: str, pattern: str, replacement: str) -> Optional[str]:
    """Apply a single SMARTS transformation to a molecule."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Parse SMARTS pattern
    patt = Chem.MolFromSmarts(pattern)
    if patt is None:
        return None

    # Check if pattern matches
    if not mol.HasSubstructMatch(patt):
        return None

    # Apply transformation
    try:
        rxn = AllChem.ReactionFromSmarts(f'{pattern}>>{replacement}')
        products = rxn.RunReactants((mol,))

        if len(products) == 0:
            return None

        # Take first product
        product_mol = products[0][0]
        Chem.SanitizeMol(product_mol)

        return Chem.MolToSmiles(product_mol)
    except:
        return None


def generate_analog_candidates(
    parent_smiles: str,
    transforms: List[Tuple],
    max_per_parent: int = 5
) -> List[Dict]:
    """
    Generate analog candidates from a single parent molecule.

    Returns list of dicts with keys:
    - smiles, parent_smiles, transform_name, mw_delta, logp_delta
    """
    candidates = []

    for pattern, replacement, name, mw_delta, logp_delta in transforms:
        analog_smiles = apply_transform(parent_smiles, pattern, replacement)

        if analog_smiles is not None and analog_smiles != parent_smiles:
            candidates.append({
                'smiles': analog_smiles,
                'parent_smiles': parent_smiles,
                'transform_name': name,
                'expected_mw_delta': mw_delta,
                'expected_logp_delta': logp_delta
            })

    # Limit candidates per parent
    if len(candidates) > max_per_parent:
        candidates = random.sample(candidates, max_per_parent)

    return candidates


def generate_diverse_analogs(
    training_smiles: List[str],
    n_target: int = 1200,
    min_diversity: float = 0.4,  # max_sim < 0.6 (diversity > 0.4)
    bias_regions: Dict = None,
    bias_target_fraction: float = 0.25,
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Generate diverse analog molecules that fill gaps in chemical space.

    Args:
        training_smiles: List of training SMILES
        n_target: Target number of analogs to generate
        min_diversity: Minimum diversity score (1 - max_sim)
        bias_regions: Dict with bias thresholds {'MW': 337, 'LogP': 4.2}
        bias_target_fraction: Target fraction in bias regions
        random_seed: Random seed for reproducibility

    Returns:
        DataFrame with columns:
        - smiles, parent_smiles, transform_name
        - max_sim_to_train (lower is better)
        - diversity_score (higher is better)
        - MW, LogP, in_mw_bias, in_logp_bias
    """
    if bias_regions is None:
        bias_regions = {'MW': 337.46, 'LogP': 4.20}

    random.seed(random_seed)
    np.random.seed(random_seed)

    print(f"Generating diverse analogs from {len(training_smiles)} training molecules...")

    # Precompute training fingerprints for similarity calculations
    print("Computing training fingerprints...")
    training_fps = []
    for smiles in training_smiles:
        fp = get_morgan_fp(smiles)
        if fp is not None:
            training_fps.append(fp)

    print(f"  {len(training_fps)} valid training fingerprints")

    # Get parent descriptors to prioritize parents near bias thresholds
    print("Analyzing parent molecules...")
    parent_data = []
    for smiles in training_smiles:
        desc = get_molecular_descriptors(smiles)
        if desc is not None:
            # Distance to bias thresholds
            mw_dist = abs(desc['MW'] - bias_regions['MW'])
            logp_dist = abs(desc['LogP'] - bias_regions['LogP'])

            # Prioritize parents close to thresholds (can be pushed into bias region)
            priority = 1.0 / (1.0 + mw_dist/100 + logp_dist)

            parent_data.append({
                'smiles': smiles,
                'MW': desc['MW'],
                'LogP': desc['LogP'],
                'priority': priority
            })

    parent_df = pd.DataFrame(parent_data)

    # Sample parents with bias toward those near thresholds
    n_parents_to_sample = min(len(parent_df), n_target // 3)  # 3-4 analogs per parent

    # 70% from high-priority parents, 30% random
    n_priority = int(n_parents_to_sample * 0.7)
    n_random = n_parents_to_sample - n_priority

    priority_parents = parent_df.nlargest(n_priority * 2, 'priority').sample(n_priority)
    random_parents = parent_df.sample(n_random)
    sampled_parents = pd.concat([priority_parents, random_parents])

    print(f"Sampled {len(sampled_parents)} parent molecules")
    print(f"  High-priority parents (near bias thresholds): {len(priority_parents)}")
    print(f"  Random parents: {len(random_parents)}")

    # Generate analogs
    print("\nGenerating analog candidates...")
    all_candidates = []

    for idx, row in sampled_parents.iterrows():
        parent_smiles = row['smiles']
        candidates = generate_analog_candidates(parent_smiles, TRANSFORMS, max_per_parent=6)
        all_candidates.extend(candidates)

    print(f"  Generated {len(all_candidates)} candidate analogs")

    # Deduplicate
    candidates_df = pd.DataFrame(all_candidates)
    candidates_df = candidates_df.drop_duplicates(subset='smiles')
    print(f"  {len(candidates_df)} unique candidates after deduplication")

    # Compute descriptors and diversity scores
    print("\nComputing diversity scores...")
    scored_candidates = []

    for idx, row in candidates_df.iterrows():
        smiles = row['smiles']

        # Get descriptors
        desc = get_molecular_descriptors(smiles)
        if desc is None:
            continue

        # Compute similarity to training
        max_sim = compute_max_similarity_to_set(smiles, training_fps)
        diversity_score = 1.0 - max_sim

        # Check if in bias regions
        in_mw_bias = desc['MW'] > bias_regions['MW']
        in_logp_bias = desc['LogP'] > bias_regions['LogP']
        in_any_bias = in_mw_bias or in_logp_bias

        # Bias bonus: 2x score if in bias region AND diverse
        bias_bonus = 2.0 if in_any_bias else 1.0

        # Combined score: diversity * bias_bonus
        combined_score = diversity_score * bias_bonus

        scored_candidates.append({
            'smiles': smiles,
            'parent_smiles': row['parent_smiles'],
            'transform_name': row['transform_name'],
            'max_sim_to_train': max_sim,
            'diversity_score': diversity_score,
            'combined_score': combined_score,
            'MW': desc['MW'],
            'LogP': desc['LogP'],
            'TPSA': desc['TPSA'],
            'NumRotatableBonds': desc['NumRotatableBonds'],
            'in_mw_bias': in_mw_bias,
            'in_logp_bias': in_logp_bias,
            'in_any_bias': in_any_bias
        })

    scored_df = pd.DataFrame(scored_candidates)

    # Filter by minimum diversity
    diverse_df = scored_df[scored_df['diversity_score'] >= min_diversity].copy()
    print(f"  {len(diverse_df)} candidates with diversity >= {min_diversity}")

    # Select top n_target by combined score
    if len(diverse_df) > n_target:
        selected_df = diverse_df.nlargest(n_target, 'combined_score')
    else:
        selected_df = diverse_df

    print(f"\nSelected {len(selected_df)} candidates")

    # Report statistics
    bias_fraction = selected_df['in_any_bias'].mean()
    mw_bias_fraction = selected_df['in_mw_bias'].mean()
    logp_bias_fraction = selected_df['in_logp_bias'].mean()

    print(f"\nBias region coverage:")
    print(f"  Any bias region: {bias_fraction:.1%} (target: {bias_target_fraction:.1%})")
    print(f"  MW > {bias_regions['MW']}: {mw_bias_fraction:.1%}")
    print(f"  LogP > {bias_regions['LogP']}: {logp_bias_fraction:.1%}")

    print(f"\nDiversity statistics:")
    print(f"  Mean diversity: {selected_df['diversity_score'].mean():.3f}")
    print(f"  Median diversity: {selected_df['diversity_score'].median():.3f}")
    print(f"  Min diversity: {selected_df['diversity_score'].min():.3f}")
    print(f"  Max diversity: {selected_df['diversity_score'].max():.3f}")

    print(f"\nSimilarity to training (max Tanimoto):")
    print(f"  Mean: {selected_df['max_sim_to_train'].mean():.3f}")
    print(f"  Median: {selected_df['max_sim_to_train'].median():.3f}")
    print(f"  Q1-Q3: {selected_df['max_sim_to_train'].quantile(0.25):.3f} - {selected_df['max_sim_to_train'].quantile(0.75):.3f}")

    return selected_df


def select_diverse_subset(
    candidates: pd.DataFrame,
    n_select: int = 600,
    diversity_weight: float = 0.6,
    bias_weight: float = 0.4,
    redundancy_threshold: float = 0.8
) -> pd.DataFrame:
    """
    Select a diverse subset from candidates using greedy diversity maximization.

    Strategy:
    1. Start with highest-scoring candidate
    2. Iteratively add candidates that are most different from already-selected
    3. Ensure bias region targets are met

    Args:
        candidates: DataFrame of candidate analogs
        n_select: Number to select
        diversity_weight: Weight for diversity in selection
        bias_weight: Weight for bias region coverage
        redundancy_threshold: Filter out candidates too similar (Tanimoto > threshold)

    Returns:
        Subset DataFrame with n_select diverse analogs
    """
    print(f"\nSelecting diverse subset of {n_select} from {len(candidates)} candidates...")

    # Compute fingerprints for all candidates
    candidate_fps = []
    valid_indices = []

    for idx, row in candidates.iterrows():
        fp = get_morgan_fp(row['smiles'])
        if fp is not None:
            candidate_fps.append(fp)
            valid_indices.append(idx)

    candidates = candidates.loc[valid_indices].copy()
    print(f"  {len(candidates)} candidates with valid fingerprints")

    # Greedy selection
    selected_indices = []
    selected_fps = []

    # Start with highest combined score
    first_idx = candidates['combined_score'].idxmax()
    selected_indices.append(first_idx)
    selected_fps.append(candidate_fps[valid_indices.index(first_idx)])

    print("  Starting greedy selection...")

    while len(selected_indices) < n_select and len(selected_indices) < len(candidates):
        # For each remaining candidate, compute minimum similarity to selected
        remaining_mask = ~candidates.index.isin(selected_indices)
        remaining_candidates = candidates[remaining_mask]

        if len(remaining_candidates) == 0:
            break

        best_score = -np.inf
        best_idx = None

        for idx, row in remaining_candidates.iterrows():
            fp_idx = valid_indices.index(idx)
            candidate_fp = candidate_fps[fp_idx]

            # Compute minimum similarity to already-selected
            if len(selected_fps) > 0:
                similarities = [DataStructs.TanimotoSimilarity(candidate_fp, sel_fp)
                               for sel_fp in selected_fps]
                min_sim_to_selected = min(similarities)
                diversity_from_selected = 1.0 - min_sim_to_selected
            else:
                diversity_from_selected = 1.0

            # Score combines:
            # 1. Diversity from training (original diversity_score)
            # 2. Diversity from already-selected
            # 3. Bias region bonus
            score = (diversity_weight * diversity_from_selected +
                    (1 - diversity_weight) * row['diversity_score'])

            if row['in_any_bias']:
                score *= (1 + bias_weight)

            if score > best_score:
                best_score = score
                best_idx = idx

        if best_idx is not None:
            selected_indices.append(best_idx)
            selected_fps.append(candidate_fps[valid_indices.index(best_idx)])

        # Progress reporting
        if len(selected_indices) % 100 == 0:
            print(f"    Selected {len(selected_indices)}/{n_select}...")

    selected_df = candidates.loc[selected_indices].copy()

    print(f"\nSelected {len(selected_df)} diverse analogs")
    print(f"  Mean diversity: {selected_df['diversity_score'].mean():.3f}")
    print(f"  Bias coverage: {selected_df['in_any_bias'].mean():.1%}")

    return selected_df


def generate_and_select_synthetics(
    training_df: pd.DataFrame,
    n_candidates: int = 1200,
    n_final: int = 600,
    min_diversity: float = 0.4,
    bias_regions: Dict = None,
    output_prefix: str = 'memory/esol_synthetics'
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Complete pipeline: generate candidates and select diverse subset.

    Args:
        training_df: Training data with 'smiles' column
        n_candidates: Number of candidates to generate
        n_final: Number to select for final set
        min_diversity: Minimum diversity threshold
        bias_regions: Bias region thresholds
        output_prefix: Prefix for output files

    Returns:
        (candidates_df, selected_df)
    """
    training_smiles = training_df['smiles'].tolist()

    # Generate candidates
    candidates_df = generate_diverse_analogs(
        training_smiles=training_smiles,
        n_target=n_candidates,
        min_diversity=min_diversity,
        bias_regions=bias_regions
    )

    # Save candidates
    candidates_df.to_csv(f'{output_prefix}_candidates.csv', index=False)
    print(f"\n✅ Saved {len(candidates_df)} candidates to {output_prefix}_candidates.csv")

    # Select diverse subset
    selected_df = select_diverse_subset(
        candidates=candidates_df,
        n_select=n_final
    )

    # Save selected
    selected_df.to_csv(f'{output_prefix}_filtered.csv', index=False)
    print(f"✅ Saved {len(selected_df)} selected analogs to {output_prefix}_filtered.csv")

    return candidates_df, selected_df
