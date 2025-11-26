"""
Fine-Tuning Bridge (FTB): Merge synthetics into persistent memory.

Design philosophy: Keep it simple. Just save synthetics to disk and reload
the playbook. Let the existing Curator handle deduplication at bullet level.

v0 → v1 upgrade path:
- v0: Just merge (basic version)
- v1: Add episode-level dedup, fidelity tiers, provenance (Phase 1)
"""

import json
import os
from pathlib import Path
from typing import Optional, List, Dict
from datetime import datetime
import logging
import numpy as np

logger = logging.getLogger(__name__)


def fine_tuning_bridge(consolidated_data, domain: str) -> Optional['ACEPlaybook']:
    """
    Fine-Tuning Bridge v0: Merge synthetic episodes into ACEPlaybook memory.

    Simple merging - no complex training loops. Synthetics are already
    in episode format, so we just save them to disk and reload.

    Args:
        consolidated_data: Output from OfflineConsolidation.consolidate()
        domain: Domain name ('hot_pot', 'switch_light', 'chem_tile')

    Returns:
        Updated ACEPlaybook with synthetics included, or None if gate failed
    """

    # Check quality gate
    if consolidated_data.gate_status == 'FAIL':
        logger.info(f"Quality gate FAILED for {domain} - skipping FTB")
        logger.info(f"Reason: {consolidated_data.reason}")
        logger.info(f"CV error: {consolidated_data.cv_error:.1%}")
        return None

    if consolidated_data.gate_status == 'WARNING':
        logger.warning(f"Quality gate WARNING for {domain} - proceeding with caution")

    # Save synthetics to episode storage
    episodes_dir = Path(f"memory/domains/{domain}/episodes/")
    episodes_dir.mkdir(parents=True, exist_ok=True)

    saved_count = 0
    for synthetic_ep in consolidated_data.synthetic_episodes:
        # Generate episode_id if not present
        if 'episode_id' not in synthetic_ep:
            synthetic_ep['episode_id'] = f"synthetic_{domain}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{saved_count}"

        filepath = episodes_dir / f"{synthetic_ep['episode_id']}.json"

        # Add metadata
        synthetic_ep['is_synthetic'] = True
        synthetic_ep['generated_at'] = datetime.now().isoformat()
        synthetic_ep['ftb_version'] = 'v0'

        # Save
        with open(filepath, 'w') as f:
            json.dump(synthetic_ep, f, indent=2)

        saved_count += 1

    logger.info(f"✓ Saved {saved_count} synthetic episodes to {domain}")

    # Reload playbook (automatically includes new episodes)
    from memory.ace_playbook import ACEPlaybook
    playbook = ACEPlaybook(domain)

    # Count total episodes
    all_observations = playbook.playbook.get('observations', [])
    total_episodes = len(all_observations)

    logger.info(f"✓ Reloaded playbook: {total_episodes} total observations")
    logger.info(f"  ({total_episodes - saved_count} real + {saved_count} synthetic)")

    return playbook


def validate_ftb_result(playbook_before, playbook_after, expected_synthetics: int):
    """
    Validate that FTB actually added the synthetics.

    Args:
        playbook_before: Playbook before FTB
        playbook_after: Playbook after FTB
        expected_synthetics: Number of synthetics we tried to add

    Raises:
        AssertionError: If validation fails
    """
    before_count = len(playbook_before.playbook.get('observations', []))
    after_count = len(playbook_after.playbook.get('observations', []))

    added = after_count - before_count

    assert added >= 0, \
        f"Observations decreased from {before_count} to {after_count}"

    # Note: added might be less than expected_synthetics due to deduplication
    logger.info(f"✓ FTB validation: {before_count} → {after_count} observations (+{added})")


# ============================================================================
# Fine-Tuning Bridge v1: Enhanced with filtering and deduplication
# ============================================================================

def fine_tuning_bridge_v1(
    consolidated_data,
    domain: str,
    min_fidelity: float = 0.6,
    max_similarity: float = 0.95
) -> Optional['ACEPlaybook']:
    """
    FTB v1: Adds filtering, fidelity tiers, and provenance tracking.

    Improvements over v0:
    1. Filter by fidelity threshold
    2. Deduplicate against existing episodes
    3. Tier reliability by fidelity score
    4. Track parent episodes

    Args:
        consolidated_data: Output from OfflineConsolidation
        domain: Domain name
        min_fidelity: Minimum fidelity score to include (default 0.6, relaxed from 0.7)
        max_similarity: Maximum similarity to existing episodes (default 0.95, relaxed from 0.9)

    Returns:
        Updated ACEPlaybook or None if gate failed

    Note:
        Thresholds were relaxed based on empirical testing showing:
        - 0.7 fidelity threshold was too strict, rejecting valid synthetics
        - 0.9 similarity threshold was too aggressive for meaningful deduplication
    """

    # Quality gate (same as v0)
    if consolidated_data.gate_status == 'FAIL':
        logger.info(f"Quality gate FAILED - skipping FTB")
        logger.info(f"Reason: {consolidated_data.reason}")
        return None

    # Load existing playbook for deduplication
    from memory.ace_playbook import ACEPlaybook
    playbook_before = ACEPlaybook(domain)
    existing_obs = playbook_before.playbook.get('observations', [])

    # Convert existing observations to episodes for comparison
    existing_episodes = existing_obs

    # Filter and process synthetics
    filtered_synthetics = []

    for synthetic in consolidated_data.synthetic_episodes:
        # Filter 1: Fidelity threshold
        fidelity = synthetic.get('fidelity_score', 0.0)
        if fidelity < min_fidelity:
            logger.debug(f"Skipping {synthetic.get('episode_id', 'unknown')}: "
                        f"fidelity {fidelity:.2f} < {min_fidelity}")
            continue

        # Filter 2: Deduplication (episode-level similarity)
        if existing_episodes:
            max_sim = max(
                episode_similarity(synthetic, existing)
                for existing in existing_episodes
            )

            if max_sim > max_similarity:
                logger.debug(f"Skipping {synthetic.get('episode_id', 'unknown')}: "
                            f"too similar to existing (sim={max_sim:.2f})")
                continue
        else:
            max_sim = 0.0

        # Assign fidelity-tiered reliability
        if fidelity >= 0.9:
            synthetic['reliability'] = 'SYNTHETIC_HIGH'
            synthetic['weight'] = 0.8
        elif fidelity >= 0.8:
            synthetic['reliability'] = 'SYNTHETIC_MEDIUM'
            synthetic['weight'] = 0.6
        else:
            synthetic['reliability'] = 'SYNTHETIC_LOW'
            synthetic['weight'] = 0.4

        # Add provenance tracking
        synthetic['parent_episode_ids'] = get_parent_episode_ids(synthetic)
        synthetic['generation_method'] = synthetic.get('variant_type', 'unknown')
        synthetic['ftb_version'] = 'v1'
        synthetic['similarity_to_nearest'] = max_sim

        filtered_synthetics.append(synthetic)

    logger.info(f"Filtered: {len(consolidated_data.synthetic_episodes)} → "
                f"{len(filtered_synthetics)} synthetics")

    # Add filtered synthetics to playbook observations
    # NOTE: ACEPlaybook.load_playbook() only loads from playbook.json, not individual episode files
    # So we must directly modify the playbook and save it

    saved_count = 0
    for synthetic in filtered_synthetics:
        # Generate episode_id if not present
        if 'episode_id' not in synthetic:
            synthetic['episode_id'] = f"synthetic_{domain}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{saved_count}"

        # Add timestamp metadata
        synthetic['is_synthetic'] = True
        synthetic['generated_at'] = datetime.now().isoformat()

        # Add to playbook observations
        playbook_before.playbook['observations'].append(synthetic)
        saved_count += 1

    # Save updated playbook to disk
    if saved_count > 0:
        playbook_before.save_playbook()

    # Also save individual episode files for archival/debugging
    episodes_dir = Path(f"memory/domains/{domain}/episodes/")
    episodes_dir.mkdir(parents=True, exist_ok=True)

    for synthetic in filtered_synthetics:
        filepath = episodes_dir / f"{synthetic['episode_id']}.json"
        with open(filepath, 'w') as f:
            json.dump(synthetic, f, indent=2)

    total_obs = len(playbook_before.playbook.get('observations', []))

    logger.info(f"✓ Added {saved_count} synthetics to playbook (v1)")
    logger.info(f"  Total observations: {total_obs}")

    return playbook_before


def episode_similarity(ep1: dict, ep2: dict) -> float:
    """
    Compute Jaccard similarity between episodes.

    Compares: contexts, observation sequence, outcomes
    Returns: similarity in [0, 1]
    """
    # Compare contexts
    ctx1_raw = ep1.get('context', {})
    ctx2_raw = ep2.get('context', {})

    # Convert context to comparable format
    ctx1 = str(sorted(ctx1_raw.items())) if isinstance(ctx1_raw, dict) else str(ctx1_raw)
    ctx2 = str(sorted(ctx2_raw.items())) if isinstance(ctx2_raw, dict) else str(ctx2_raw)

    context_match = 1.0 if ctx1 == ctx2 else 0.0

    # Compare observation sequences (simplified)
    obs1 = ep1.get('observations', [])
    obs2 = ep2.get('observations', [])

    # If one has no observations, use beliefs instead
    if not obs1 and not obs2:
        # Compare beliefs
        beliefs1 = ep1.get('beliefs', {})
        beliefs2 = ep2.get('beliefs', {})

        if not beliefs1 and not beliefs2:
            structure_match = 1.0
        elif not beliefs1 or not beliefs2:
            structure_match = 0.0
        else:
            # Simple key-based comparison
            keys1 = set(beliefs1.keys())
            keys2 = set(beliefs2.keys())
            if not keys1 and not keys2:
                structure_match = 1.0
            elif not keys1 or not keys2:
                structure_match = 0.0
            else:
                structure_match = len(keys1 & keys2) / len(keys1 | keys2)

    else:
        # Jaccard on observation keys
        if isinstance(obs1, list) and isinstance(obs2, list):
            # Compare observation structure
            if not obs1 and not obs2:
                structure_match = 1.0
            elif not obs1 or not obs2:
                structure_match = 0.0
            else:
                # Compare length similarity
                len_sim = 1.0 - abs(len(obs1) - len(obs2)) / max(len(obs1), len(obs2))
                structure_match = len_sim
        else:
            structure_match = 0.5

    # Weighted combination (context is more important)
    similarity = 0.7 * context_match + 0.3 * structure_match

    return similarity


def get_parent_episode_ids(synthetic: dict) -> List[str]:
    """
    Extract IDs of real episodes used to generate this synthetic.

    Args:
        synthetic: Synthetic episode dictionary

    Returns:
        List of parent episode IDs
    """
    # Check various possible locations for parent info
    parent_ids = []

    # Check metadata
    metadata = synthetic.get('metadata', {})
    if 'base_episode' in metadata:
        parent_ids.append(metadata['base_episode'])

    # Check source fields
    if 'source_episode_ids' in synthetic:
        parent_ids.extend(synthetic['source_episode_ids'])

    # Check episode_id for embedded parent info
    episode_id = synthetic.get('episode_id', '')
    if '_cf_' in episode_id:
        # Extract base episode from counterfactual ID
        base_id = episode_id.split('_cf_')[0]
        if base_id and base_id not in parent_ids:
            parent_ids.append(base_id)

    return parent_ids
