"""
Phase 1b Diagnostic Script
Run this to identify implementation issues in the stress test.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from molecular_design_env import MolecularDesignEnv, DataCorruptor
from molecular_world_model import MolecularWorldModel, load_esol_data

def print_section(title):
    print(f"\n{'='*70}")
    print(f" {title}")
    print('='*70)

def diagnose():
    data = load_esol_data('data/esol_processed.pkl')
    candidate_pool = data['candidate_pool']
    test_set = data['test_set']

    # =========================================================================
    # DIAGNOSTIC 1: DataCorruptor Implementation Check
    # =========================================================================
    print_section("DIAGNOSTIC 1: DataCorruptor Class Inspection")

    # Check what modes are implemented
    print("\n1a. Checking DataCorruptor modes...")

    for mode in ['none', 'noise', 'shift', 'imbalance', 'drift']:
        corruptor = DataCorruptor(mode, severity=0.3, seed=42)
        print(f"\n  Mode: '{mode}'")
        print(f"    - corruptor.mode = {corruptor.mode}")

        # Check if corrupt_label method exists and what it does
        if hasattr(corruptor, 'corrupt_label'):
            # Test with a sample label
            true_label = -2.0
            results = []
            for step in range(50):
                corruptor_fresh = DataCorruptor(mode, severity=0.3, noise_magnitude=0.5,
                                                drift_offset=0.3, drift_threshold=25, seed=42+step)
                corrupted = corruptor_fresh.corrupt_label(true_label, step=step, context=(0, 1, 2))
                if abs(corrupted - true_label) > 1e-6:
                    results.append((step, corrupted))

            if results:
                print(f"    - corrupt_label DOES modify labels ({len(results)} changes in 50 steps)")
                print(f"    - Example changes: {results[:5]}")
            else:
                print(f"    - corrupt_label does NOT modify labels (0 changes in 50 steps)")
        else:
            print(f"    - WARNING: corrupt_label method not found!")

        # Check filter_candidates if it exists
        if hasattr(corruptor, 'filter_candidates'):
            print(f"    - filter_candidates method exists")
        else:
            print(f"    - filter_candidates method NOT found")

    # =========================================================================
    # DIAGNOSTIC 2: Environment Data Flow
    # =========================================================================
    print_section("DIAGNOSTIC 2: Environment Data Flow")

    world_model = MolecularWorldModel(random_state=42)

    # Test with noise corruptor
    corruptor = DataCorruptor('noise', severity=0.30, noise_magnitude=0.5, seed=42)
    env = MolecularDesignEnv(
        candidate_pool=candidate_pool,
        test_set=test_set,
        world_model=world_model,
        query_budget=50,
        random_state=42,
        corruptor=corruptor
    )

    state = env.reset(seed_size=5)

    print("\n2a. Checking what env.oracle contains...")
    print(f"    - Type: {type(env.oracle)}")
    print(f"    - Length: {len(env.oracle) if hasattr(env.oracle, '__len__') else 'N/A'}")
    print(f"    - First 5 values: {[env.oracle[i] for i in range(5)]}")

    # Check if oracle is the true labels or something else
    true_labels = candidate_pool['logS'].values
    oracle_matches_true = np.allclose(
        [env.oracle[i] for i in range(min(10, len(true_labels)))],
        true_labels[:10]
    )
    print(f"    - Oracle matches true labels? {oracle_matches_true}")

    print("\n2b. Tracing a single step with 30% noise...")

    # Get unqueried indices
    unqueried = [i for i in range(len(env.all_smiles)) if i not in env.queried_indices]
    action_idx = unqueried[0]
    true_label_for_action = true_labels[action_idx]

    print(f"    - Action index: {action_idx}")
    print(f"    - True label (from candidate_pool): {true_label_for_action:.4f}")
    print(f"    - env.oracle[{action_idx}]: {env.oracle[action_idx]:.4f}")

    obs, reward, done, info = env.step(action_idx)

    print(f"    - After step:")
    print(f"      - info keys: {list(info.keys())}")
    if 'observed_label' in info:
        print(f"      - info['observed_label']: {info['observed_label']:.4f}")
    if 'true_label' in info:
        print(f"      - info['true_label']: {info['true_label']:.4f}")
    if 'corrupted' in info:
        print(f"      - info['corrupted']: {info['corrupted']}")

    # Check observation content
    print(f"    - obs keys: {list(obs.keys())}")
    if 'query_result' in obs:
        qr = obs['query_result']
        print(f"      - query_result keys: {list(qr.keys())}")
        if 'true_label' in qr:
            print(f"      - query_result['true_label']: {qr['true_label']:.4f}")

    # Check what the episode records
    if hasattr(env, 'episode') and env.episode is not None:
        print(f"\n2c. What does env.episode contain?")
        episode = env.episode
        print(f"    - Episode type: {type(episode)}")
        print(f"    - Episode attributes: {[a for a in dir(episode) if not a.startswith('_')]}")

        if hasattr(episode, 'steps') and episode.steps:
            print(f"    - Number of steps: {len(episode.steps)}")
            print(f"    - First step keys: {list(episode.steps[0].keys())}")
            step0 = episode.steps[0]
            for key in ['true_label', 'corrupted_label', 'was_corrupted', 'prediction', 'error']:
                if key in step0:
                    print(f"      - steps[0]['{key}']: {step0[key]}")

    # =========================================================================
    # DIAGNOSTIC 3: What Do Online vs Full Stack Models Actually See?
    # =========================================================================
    print_section("DIAGNOSTIC 3: Online vs Full Stack Data Comparison")

    # Reset and run 20 steps
    corruptor = DataCorruptor('noise', severity=0.30, noise_magnitude=0.5, seed=42)
    env = MolecularDesignEnv(
        candidate_pool=candidate_pool,
        test_set=test_set,
        world_model=world_model,
        query_budget=50,
        random_state=42,
        corruptor=corruptor
    )
    state = env.reset(seed_size=5)

    # Run 20 steps
    for _ in range(20):
        unqueried = [i for i in range(len(env.all_smiles)) if i not in env.queried_indices]
        if not unqueried:
            break
        action = unqueried[0]
        env.step(action)

    print("\n3a. What OnlineModel.update() would use:")
    queried_indices = list(env.queried_indices)
    online_labels = [env.oracle[i] for i in queried_indices]
    true_labels_subset = [true_labels[i] for i in queried_indices]

    print(f"    - Number of queried points: {len(queried_indices)}")
    print(f"    - Labels from env.oracle (first 5): {online_labels[:5]}")
    print(f"    - True labels (first 5): {true_labels_subset[:5]}")

    # Check if they're identical
    differences = [(i, ol, tl) for i, (ol, tl) in enumerate(zip(online_labels, true_labels_subset))
                   if abs(ol - tl) > 1e-6]
    print(f"    - Differences found: {len(differences)}")
    if differences:
        print(f"    - First 3 differences: {differences[:3]}")

    print("\n3b. What FullStackModel.update() would use (via env.get_episode()):")
    episode = env.get_episode()
    print(f"    - Episode type: {type(episode)}")

    if isinstance(episode, dict):
        print(f"    - Episode keys: {list(episode.keys())}")
        if 'steps' in episode:
            print(f"    - Number of steps in episode: {len(episode['steps'])}")
            if episode['steps']:
                step_keys = list(episode['steps'][0].keys())
                print(f"    - Step keys: {step_keys}")

                # Extract labels from steps
                episode_labels = []
                episode_true_labels = []
                episode_corrupted_labels = []

                for step in episode['steps']:
                    if 'true_label' in step:
                        episode_true_labels.append(step['true_label'])
                    if 'corrupted_label' in step:
                        episode_corrupted_labels.append(step['corrupted_label'])

                if episode_true_labels:
                    print(f"    - episode steps 'true_label' (first 5): {episode_true_labels[:5]}")
                if episode_corrupted_labels:
                    print(f"    - episode steps 'corrupted_label' (first 5): {episode_corrupted_labels[:5]}")

                    # Check how many were actually corrupted
                    n_corrupted = sum(1 for t, c in zip(episode_true_labels, episode_corrupted_labels)
                                     if abs(t - c) > 1e-6)
                    print(f"    - Number of corrupted labels in episode: {n_corrupted}/{len(episode_true_labels)}")

    # =========================================================================
    # DIAGNOSTIC 4: Corruption Statistics
    # =========================================================================
    print_section("DIAGNOSTIC 4: Corruption Statistics")

    print("\n4a. Running 50 steps with each corruption mode and counting actual corruptions...")

    for mode in ['none', 'noise', 'shift', 'imbalance', 'drift']:
        corruptor = DataCorruptor(mode, severity=0.30, noise_magnitude=0.5,
                                  drift_offset=0.3, drift_threshold=25, seed=42)

        world_model = MolecularWorldModel(random_state=42)
        env = MolecularDesignEnv(
            candidate_pool=candidate_pool,
            test_set=test_set,
            world_model=world_model,
            query_budget=50,
            random_state=42,
            corruptor=corruptor
        )
        state = env.reset(seed_size=5)

        # Fit initial model
        seed_indices = list(env.queried_indices)
        seed_smiles = [env.all_smiles[i] for i in seed_indices]
        seed_labels = [env.oracle[i] for i in seed_indices]
        world_model.fit(seed_smiles, seed_labels)

        # Run 45 steps (50 budget - 5 seed)
        step_data = []
        for step_num in range(45):
            unqueried = [i for i in range(len(env.all_smiles)) if i not in env.queried_indices]
            if not unqueried:
                break
            action = unqueried[0]
            obs, reward, done, info = env.step(action)

            # Get the true label and what was recorded
            true_label = true_labels[action]

            step_info = {
                'step': step_num,
                'action': action,
                'true_label': true_label,
                'oracle_label': env.oracle[action]
            }

            # Check episode step
            if env.episode and env.episode.steps:
                last_step = env.episode.steps[-1]
                step_info['episode_true_label'] = last_step.get('true_label')
                step_info['episode_corrupted_label'] = last_step.get('corrupted_label')
                step_info['episode_was_corrupted'] = last_step.get('was_corrupted')

            step_data.append(step_info)

        # Analyze
        stats = corruptor.get_stats() if hasattr(corruptor, 'get_stats') else {}

        # Count actual corruptions from episode data
        n_corrupted_in_episode = 0
        if step_data and 'episode_corrupted_label' in step_data[0]:
            for sd in step_data:
                if sd.get('episode_true_label') is not None and sd.get('episode_corrupted_label') is not None:
                    if abs(sd['episode_true_label'] - sd['episode_corrupted_label']) > 1e-6:
                        n_corrupted_in_episode += 1

        print(f"\n  Mode '{mode}':")
        print(f"    - corruptor.get_stats(): {stats}")
        print(f"    - Corruptions detected in episode data: {n_corrupted_in_episode}/{len(step_data)}")

        # Check if drift mode corrupted after threshold
        if mode == 'drift':
            post_threshold = [sd for sd in step_data if sd['step'] > 25]
            n_post_threshold_corrupted = sum(1 for sd in post_threshold
                                             if sd.get('episode_was_corrupted', False))
            print(f"    - Post-threshold (step>25) corruptions: {n_post_threshold_corrupted}/{len(post_threshold)}")

    # =========================================================================
    # DIAGNOSTIC 5: OC Gate Behavior
    # =========================================================================
    print_section("DIAGNOSTIC 5: OC Gate Analysis")

    from molecular_consolidation_pipeline import MolecularConsolidationPipeline

    print("\n5a. Testing gate with clean data...")

    world_model = MolecularWorldModel(random_state=42)
    corruptor = DataCorruptor('none', seed=42)
    env = MolecularDesignEnv(
        candidate_pool=candidate_pool,
        test_set=test_set,
        world_model=world_model,
        query_budget=50,
        random_state=42,
        corruptor=corruptor
    )

    state = env.reset(seed_size=10)

    # Fit initial model
    seed_indices = list(env.queried_indices)
    seed_smiles = [env.all_smiles[i] for i in seed_indices]
    seed_labels = [env.oracle[i] for i in seed_indices]
    world_model.fit(seed_smiles, seed_labels)

    # Run 20 more steps
    for _ in range(20):
        unqueried = [i for i in range(len(env.all_smiles)) if i not in env.queried_indices]
        if not unqueried:
            break
        action = unqueried[0]
        env.step(action)

    # Try consolidation
    test_smiles_list = test_set['smiles'].tolist()
    test_labels_list = test_set['logS'].tolist()

    pipeline = MolecularConsolidationPipeline(
        world_model=world_model,
        test_smiles=test_smiles_list,
        test_labels=test_labels_list,
        probe_size=50,
        retention_threshold=0.25,
        cv_threshold=0.20,
        output_dir='results/diagnostic',
        random_state=42
    )

    episode = env.get_episode()
    result = pipeline.consolidate([episode])

    print(f"    - Gate status: {result.gate_status}")
    print(f"    - FTB triggered: {result.ftb_triggered}")
    if hasattr(result, 'rejection_reasons'):
        print(f"    - Rejection reasons: {result.rejection_reasons}")
    if hasattr(result, 'gate_details'):
        print(f"    - Gate details: {result.gate_details}")

    # Print all attributes of result
    print(f"    - Result attributes: {[a for a in dir(result) if not a.startswith('_')]}")
    for attr in dir(result):
        if not attr.startswith('_'):
            val = getattr(result, attr)
            if not callable(val):
                print(f"      - result.{attr}: {val}")

    # Check what the CV threshold is doing
    print(f"\n5b. CV threshold analysis:")
    print(f"    - Configured CV threshold: 0.20")

    # Get predictions and uncertainties for queried points
    queried_smiles = [env.all_smiles[i] for i in env.queried_indices]
    preds, uncerts = world_model.predict(queried_smiles, return_uncertainty=True)

    if len(preds) > 0:
        mean_abs_pred = np.mean(np.abs(preds))
        std_pred = np.std(preds)
        if mean_abs_pred > 0:
            cv = std_pred / mean_abs_pred
            print(f"    - Mean absolute prediction: {mean_abs_pred:.4f}")
            print(f"    - Std of predictions: {std_pred:.4f}")
            print(f"    - CV of predictions: {cv:.4f}")
            print(f"    - Would pass CV gate (cv < 0.20)? {cv < 0.20}")

        # Also compute MAE
        queried_labels = [env.oracle[i] for i in env.queried_indices]
        mae = np.mean(np.abs(np.array(preds) - np.array(queried_labels)))
        print(f"    - MAE on queried data: {mae:.4f}")

    # =========================================================================
    # DIAGNOSTIC 6: Why distribution_shift and context_imbalance are identical
    # =========================================================================
    print_section("DIAGNOSTIC 6: distribution_shift and context_imbalance Analysis")

    print("\n6a. Checking if filter_candidates is being called...")

    for mode in ['shift', 'imbalance']:
        corruptor = DataCorruptor(mode, severity=0.5, seed=42)

        # Create some sample contexts
        sample_contexts = [(i % 20, i % 5, i % 5) for i in range(100)]
        sample_indices = list(range(100))

        filtered = corruptor.filter_candidates(sample_indices, sample_contexts)

        print(f"\n  Mode '{mode}':")
        print(f"    - Original indices: {len(sample_indices)}")
        print(f"    - After filter_candidates: {len(filtered)}")
        print(f"    - Reduction: {len(sample_indices) - len(filtered)}")

        if mode == 'shift':
            # Check if low-MW molecules are selected
            low_mw = [i for i, ctx in zip(sample_indices, sample_contexts) if ctx[1] < 2]
            print(f"    - Low MW (mw_bin < 2) in original: {len(low_mw)}")

        if mode == 'imbalance':
            # Check cluster 0 ratio
            cluster_0 = [i for i, ctx in zip(sample_indices, sample_contexts) if ctx[0] == 0]
            filtered_cluster_0 = [i for i in filtered if sample_contexts[i][0] == 0]
            print(f"    - Cluster 0 in original: {len(cluster_0)}")
            print(f"    - Cluster 0 in filtered: {len([f for f in filtered if f < len(sample_contexts) and sample_contexts[f][0] == 0])}")

    print("\n6b. Checking if filter_candidates is called in MolecularDesignEnv...")

    # Search for filter_candidates usage in env
    import inspect
    env_source = inspect.getsource(MolecularDesignEnv)

    if 'filter_candidates' in env_source:
        print("    - 'filter_candidates' IS referenced in MolecularDesignEnv source")
        # Find the context
        for i, line in enumerate(env_source.split('\n')):
            if 'filter_candidates' in line:
                print(f"      Line {i}: {line.strip()}")
    else:
        print("    - 'filter_candidates' is NOT referenced in MolecularDesignEnv source")
        print("    - BUG FOUND: Corruption modes 'shift' and 'imbalance' filter_candidates")
        print("      is implemented but never called!")

    # =========================================================================
    # DIAGNOSTIC 7: Source Code Key Sections
    # =========================================================================
    print_section("DIAGNOSTIC 7: Key Source Code Review")

    print("\n7a. DataCorruptor.corrupt_label() method:")
    if hasattr(DataCorruptor, 'corrupt_label'):
        print(inspect.getsource(DataCorruptor.corrupt_label))

    print("\n7b. Where is corrupt_label called in env.step()?")
    step_source = inspect.getsource(MolecularDesignEnv.step)
    for i, line in enumerate(step_source.split('\n')):
        if 'corrupt' in line.lower():
            print(f"  Line {i}: {line}")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print_section("DIAGNOSTIC SUMMARY")
    print("""
    Review the output above for these specific issues:

    1. CORRUPTION FLOW: Is corrupt_label() actually being called in env.step()?
       Does it modify labels for 'noise', 'shift', 'imbalance', 'drift' modes?

    2. ONLINE MODEL DATA: Does OnlineModel see corrupted or true labels?
       Check if env.oracle contains true labels and Online is bypassing corruption.

    3. IDENTICAL CONDITIONS: Why are distribution_shift and context_imbalance
       producing identical results to clean? Are filter_candidates being called?

    4. GATE BEHAVIOR: Why 0% pass rate? Is CV threshold (0.20) too strict?
       What are the actual CV values?

    5. EPISODE DATA: Does get_episode() return corrupted or true labels?
       This determines what Full Stack model trains on.

    KEY QUESTIONS ANSWERED:
    - Does OnlineModel use env.oracle (true labels) or corrupted labels?
    - Does FullStackModel use episode's true_label or corrupted_label for training?
    - Is filter_candidates() ever called for shift/imbalance modes?
    """)

if __name__ == '__main__':
    diagnose()
