#!/usr/bin/env python3
"""
Comprehensive analysis of full token prediction experiment.
Generates all A1-A5 analyses, statistical tests, and summary tables.
"""

import os
import sys
import argparse
from pathlib import Path

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from evaluation.token_analysis import TokenAnalysis
import pandas as pd
import json


def main():
    parser = argparse.ArgumentParser(description="Analyze full token prediction results")
    parser.add_argument('log_dir', type=str, help='Directory containing token logs')
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for analysis results (defaults to log_dir)'
    )
    args = parser.parse_args()

    log_dir = args.log_dir
    output_dir = args.output_dir or log_dir

    print("=" * 70)
    print("COMPREHENSIVE TOKEN PREDICTION ANALYSIS")
    print("=" * 70)
    print(f"Log directory: {log_dir}")
    print(f"Output directory: {output_dir}")
    print()

    # Initialize analysis
    print("Loading token logs...")
    analysis = TokenAnalysis(log_dir)
    print(f"Loaded {len(analysis.df)} steps from {analysis.df['episode_id'].nunique()} episodes")
    print()

    # Generate summary report
    print("Generating summary report...")
    report = analysis.generate_summary_report()
    print(report)

    # Save text report
    report_path = os.path.join(output_dir, 'analysis_report.txt')
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"\n✓ Saved text report to: {report_path}")

    # === A1: COUPLING ===
    print("\n" + "=" * 70)
    print("A1: COUPLING ANALYSIS")
    print("=" * 70)

    coupling = analysis.compute_coupling()
    if len(coupling) > 0:
        coupling_path = os.path.join(output_dir, 'coupling_by_environment.csv')
        coupling.to_csv(coupling_path, index=False)
        print(f"✓ Saved coupling results to: {coupling_path}")
        print()
        print(coupling.to_string(index=False))

        # Coupling by agent
        coupling_agent = analysis.compute_coupling_by_agent()
        coupling_agent_path = os.path.join(output_dir, 'coupling_by_agent.csv')
        coupling_agent.to_csv(coupling_agent_path, index=False)
        print(f"\n✓ Saved agent-stratified coupling to: {coupling_agent_path}")
    else:
        print("⚠ No coupling data available")

    # === A2: SURPRISE DETECTION ===
    print("\n" + "=" * 70)
    print("A2: SURPRISE DETECTION")
    print("=" * 70)

    surprise = analysis.compute_surprise_detection(surprisal_threshold=2.0)
    if len(surprise) > 0:
        surprise_path = os.path.join(output_dir, 'surprise_detection.csv')
        surprise.to_csv(surprise_path, index=False)
        print(f"✓ Saved surprise detection results to: {surprise_path}")
        print()
        print(surprise.to_string(index=False))
    else:
        print("⚠ No surprise detection data available")

    # === A3: PREDICTIVE VALIDITY ===
    print("\n" + "=" * 70)
    print("A3: PREDICTIVE VALIDITY")
    print("=" * 70)

    validity = analysis.compute_predictive_validity(lag=1)
    if len(validity) > 0:
        validity_path = os.path.join(output_dir, 'predictive_validity.csv')
        validity.to_csv(validity_path, index=False)
        print(f"✓ Saved predictive validity results to: {validity_path}")

        # Summary by environment
        validity_summary = validity.groupby('environment')['correlation'].agg([
            'mean', 'std', 'count'
        ]).reset_index()
        print()
        print(validity_summary.to_string(index=False))
    else:
        print("⚠ No predictive validity data available")

    # === A4: CALIBRATION ===
    print("\n" + "=" * 70)
    print("A4: CALIBRATION")
    print("=" * 70)

    calibration = analysis.compute_token_calibration()
    calib_path = os.path.join(output_dir, 'calibration_metrics.json')

    if calibration:
        with open(calib_path, 'w') as f:
            json.dump(calibration, f, indent=2)
        print(f"✓ Saved calibration metrics to: {calib_path}")
        print()
        for key, value in calibration.items():
            if isinstance(value, (int, float)):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
    else:
        print("⚠ No calibration data available")

    # === ADVANCED STATISTICAL ANALYSES ===
    print("\n" + "=" * 70)
    print("ADVANCED STATISTICAL ANALYSES")
    print("=" * 70)

    # Mutual Information (detects nonlinear dependencies)
    print("\n[1/5] Computing mutual information...")
    mi_df = analysis.compute_mutual_information()
    if len(mi_df) > 0:
        mi_path = os.path.join(output_dir, 'mutual_information.csv')
        mi_df.to_csv(mi_path, index=False)
        print(f"✓ Saved mutual information to: {mi_path}")
        print()
        print(mi_df.to_string(index=False))
    else:
        print("⚠ No mutual information data available")

    # Regression Diagnostics (polynomial fit tests)
    print("\n[2/5] Computing regression diagnostics...")
    reg_diag = analysis.compute_regression_diagnostics()
    if reg_diag:
        reg_diag_path = os.path.join(output_dir, 'regression_diagnostics.json')
        with open(reg_diag_path, 'w') as f:
            json.dump(reg_diag, f, indent=2)
        print(f"✓ Saved regression diagnostics to: {reg_diag_path}")
        print()
        print(f"  R² (linear): {reg_diag.get('r_squared', 0):.4f}")
        if 'polynomial_r2' in reg_diag:
            print(f"  R² (degree 2): {reg_diag['polynomial_r2'].get(2, 0):.4f}")
            print(f"  R² (degree 3): {reg_diag['polynomial_r2'].get(3, 0):.4f}")
            print(f"  Improvement (deg 2): {reg_diag.get('improvement_deg2', 0):.4f}")
            print(f"  Improvement (deg 3): {reg_diag.get('improvement_deg3', 0):.4f}")
        if reg_diag.get('improvement_deg2', 0) > 0.1:
            print("  ⚠ Strong nonlinearity detected (deg 2 improvement > 0.1)")
    else:
        print("⚠ No regression diagnostics available")

    # Distance Correlation (comprehensive dependence measure)
    print("\n[3/5] Computing distance correlation...")
    dcor_df = analysis.compute_distance_correlation()
    if len(dcor_df) > 0:
        dcor_path = os.path.join(output_dir, 'distance_correlation.csv')
        dcor_df.to_csv(dcor_path, index=False)
        print(f"✓ Saved distance correlation to: {dcor_path}")
        print()
        print(dcor_df.to_string(index=False))
    else:
        print("⚠ No distance correlation data available (dcor library may not be installed)")

    # Agent Hierarchy Comparison
    print("\n[4/5] Comparing coupling across agent types...")
    agent_comparison = analysis.compare_agent_coupling()
    if len(agent_comparison) > 0:
        agent_comp_path = os.path.join(output_dir, 'agent_coupling_comparison.csv')
        agent_comparison.to_csv(agent_comp_path, index=False)
        print(f"✓ Saved agent coupling comparison to: {agent_comp_path}")
        print()
        print(agent_comparison.to_string(index=False))
    else:
        print("⚠ No agent comparison data available")

    # Control Experiment Comparison (if control directory provided)
    print("\n[5/5] Checking for negative control experiments...")
    control_dirs = [
        os.path.join(os.path.dirname(log_dir), 'control'),
        os.path.join(os.path.dirname(log_dir), 'negative_control'),
        'results/control',
        'results/negative_control'
    ]
    control_comparison = None
    for control_dir in control_dirs:
        if os.path.exists(control_dir) and os.path.isdir(control_dir):
            try:
                print(f"  Found control directory: {control_dir}")
                control_comparison = analysis.compare_control_coupling(control_dir)
                if len(control_comparison) > 0:
                    control_comp_path = os.path.join(output_dir, 'control_coupling_comparison.csv')
                    control_comparison.to_csv(control_comp_path, index=False)
                    print(f"✓ Saved control comparison to: {control_comp_path}")
                    print()
                    print(control_comparison.to_string(index=False))
                    break
            except Exception as e:
                print(f"  ⚠ Failed to load control from {control_dir}: {e}")

    if control_comparison is None:
        print("⚠ No negative control experiments found")
        print("  Expected location: results/control/ or results/negative_control/")
        print("  Run: python scripts/run_negative_control.py --output results/control")

    # === HYPOTHESIS TESTING ===
    print("\n" + "=" * 70)
    print("HYPOTHESIS TESTING")
    print("=" * 70)

    hypothesis_results = {}

    # H-Token1: HotPot coupling > 0.5
    if len(coupling) > 0:
        hotpot_rows = coupling[coupling['environment'].str.contains('HotPot', case=False)]
        if len(hotpot_rows) > 0:
            hotpot_r = hotpot_rows['pearson_r'].iloc[0]
            hotpot_p = hotpot_rows['pearson_p'].iloc[0]
            h1_pass = (hotpot_r > 0.5) and (hotpot_p < 0.05)
            print(f"H-Token1 (HotPot coupling > 0.5): {'✓ PASS' if h1_pass else '✗ FAIL'}")
            print(f"  Observed: r = {hotpot_r:.3f}, p = {hotpot_p:.4f}")

            hypothesis_results['H-Token1'] = {
                'description': 'HotPot coupling > 0.5',
                'result': 'PASS' if h1_pass else 'FAIL',
                'statistic': f"r={hotpot_r:.3f}, p={hotpot_p:.4f}"
            }

    # H-Token2: Actor > Observer in predictive validity
    if len(validity) > 0 and 'actor' in validity['agent_type'].values and 'observer' in validity['agent_type'].values:
        actor_validity = validity[validity['agent_type'] == 'actor']['correlation'].mean()
        observer_validity = validity[validity['agent_type'] == 'observer']['correlation'].mean()
        h2_pass = actor_validity > observer_validity
        print(f"\nH-Token2 (Actor > Observer validity): {'✓ PASS' if h2_pass else '✗ FAIL'}")
        print(f"  Actor mean r: {actor_validity:.3f}")
        print(f"  Observer mean r: {observer_validity:.3f}")

        hypothesis_results['H-Token2'] = {
            'description': 'Actor > Observer predictive validity',
            'result': 'PASS' if h2_pass else 'FAIL',
            'statistic': f"Actor={actor_validity:.3f}, Observer={observer_validity:.3f}"
        }

    # H-Token3: Expected pattern HotPot > SwitchLight > ChemTile
    if len(coupling) >= 3:
        sorted_coupling = coupling.sort_values('pearson_r', ascending=False)
        env_order = list(sorted_coupling['environment'].values)
        print(f"\nH-Token3 (Environment coupling order):")
        print(f"  Expected: HotPot > SwitchLight > ChemTile")
        print(f"  Observed: {' > '.join(env_order)}")

        hypothesis_results['H-Token3'] = {
            'description': 'Coupling strength: HotPot > SwitchLight > ChemTile',
            'result': 'See observed order',
            'observed_order': env_order
        }

    # Save hypothesis test results
    hypothesis_path = os.path.join(output_dir, 'hypothesis_tests.json')
    with open(hypothesis_path, 'w') as f:
        json.dump(hypothesis_results, f, indent=2)
    print(f"\n✓ Saved hypothesis tests to: {hypothesis_path}")

    # === FINAL SUMMARY ===
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"All results saved to: {output_dir}")
    print("\nGenerated files:")
    print("  - analysis_report.txt")
    if len(coupling) > 0:
        print("  - coupling_by_environment.csv")
        print("  - coupling_by_agent.csv")
    if len(surprise) > 0:
        print("  - surprise_detection.csv")
    if len(validity) > 0:
        print("  - predictive_validity.csv")
    if calibration:
        print("  - calibration_metrics.json")
    if len(mi_df) > 0:
        print("  - mutual_information.csv")
    if reg_diag:
        print("  - regression_diagnostics.json")
    if len(dcor_df) > 0:
        print("  - distance_correlation.csv")
    if len(agent_comparison) > 0:
        print("  - agent_coupling_comparison.csv")
    if control_comparison is not None and len(control_comparison) > 0:
        print("  - control_coupling_comparison.csv")
    if hypothesis_results:
        print("  - hypothesis_tests.json")
    print("\nNext step:")
    print(f"  python scripts/generate_token_figures.py {output_dir}")
    print("=" * 70)


if __name__ == '__main__':
    main()
