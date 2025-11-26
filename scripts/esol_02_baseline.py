"""
ESOL Baseline Experiment - MolecularWorldModel with 2-Context Scheme

Trains context-aware RandomForest models on ESOL dataset and evaluates performance.

Expected performance:
- MAE: 0.60-0.90 logS
- R²: 0.75-0.88
- Both contexts should train successfully (>400 samples each)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import json
from scipy.stats import pearsonr
from agents.molecular_world_model import MolecularWorldModel


def main():
    print("=" * 60)
    print("ESOL Baseline Experiment")
    print("=" * 60)

    # Load cleaned data
    train_df = pd.read_csv('memory/esol_train.csv')
    test_df = pd.read_csv('memory/esol_test.csv')

    print(f"\nLoaded {len(train_df)} train, {len(test_df)} test molecules")

    # Initialize model with validated hyperparameters
    model = MolecularWorldModel(
        n_estimators=100,      # Number of trees per RandomForest
        max_depth=10,          # Maximum tree depth
        min_samples_context=50 # Minimum samples to train context model
    )

    # Train on full training set
    print("\nTraining...")
    for idx, row in train_df.iterrows():
        smiles = row['smiles']
        solubility = row['solubility']

        # Update model beliefs (online learning)
        model.update_belief(smiles, solubility)

        # Progress reporting
        if (idx + 1) % 100 == 0 or idx == 0:
            stats = model.get_statistics()
            print(f"  Processed {idx + 1}/{len(train_df)} molecules")
            print(f"    Context counts: {stats['counts']}")
            print(f"    Trained models: {len(stats['trained_models'])}/2")

    # Final training stats
    stats = model.get_statistics()
    print(f"\n✅ Training complete:")
    print(f"  Total observations: {stats['total_samples']}")
    print(f"  Trained models: {len(stats['trained_models'])}/2")
    print(f"  Context counts: {stats['counts']}")

    # Evaluate on test set
    print(f"\nEvaluating on test set...")
    predictions = []
    uncertainties = []
    actuals = []
    contexts = []

    for idx, row in test_df.iterrows():
        smiles = row['smiles']
        true_solubility = row['solubility']

        # Predict
        pred, unc, ctx = model.predict_property(smiles)

        predictions.append(pred)
        uncertainties.append(unc)
        actuals.append(true_solubility)
        contexts.append(ctx)

    predictions = np.array(predictions)
    actuals = np.array(actuals)
    uncertainties = np.array(uncertainties)

    # Compute metrics
    errors = np.abs(predictions - actuals)
    mae = np.mean(errors)
    rmse = np.sqrt(np.mean((predictions - actuals) ** 2))

    # R² score
    ss_res = np.sum((actuals - predictions) ** 2)
    ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
    r2 = 1 - (ss_res / ss_tot)

    # Per-context performance
    context_performance = {}
    for ctx in set(contexts):
        ctx_mask = [c == ctx for c in contexts]
        ctx_errors = errors[ctx_mask]
        ctx_n = len(ctx_errors)
        ctx_mae = np.mean(ctx_errors)

        context_performance[str(ctx)] = {
            'mae': float(ctx_mae),
            'n_samples': int(ctx_n)
        }

    # Uncertainty calibration
    # Good uncertainty estimates should correlate with prediction errors
    if len(uncertainties) > 0 and np.std(uncertainties) > 0:
        corr, p_value = pearsonr(uncertainties, errors)
    else:
        corr, p_value = 0.0, 1.0

    # Print results
    print("\n" + "=" * 60)
    print("BASELINE RESULTS")
    print("=" * 60)
    print(f"MAE:  {mae:.3f} logS")
    print(f"RMSE: {rmse:.3f} logS")
    print(f"R²:   {r2:.3f}")

    print(f"\nPer-context performance:")
    for ctx, perf in sorted(context_performance.items()):
        print(f"  {ctx}: MAE={perf['mae']:.3f} (n={perf['n_samples']})")

    print(f"\nUncertainty calibration:")
    print(f"  Correlation(uncertainty, |error|): r={corr:.3f}, p={p_value:.4f}")

    # Interpretation
    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)

    if mae < 0.6:
        print("🎉 Excellent performance (MAE < 0.6)")
        print("   This is better than expected - ESOL may be easier than anticipated")
    elif mae <= 0.9:
        print("✅ Good performance (MAE 0.6-0.9)")
        print("   Performance is within expected range")
    elif mae <= 1.2:
        print("⚠️  Below target (MAE 0.9-1.2)")
        print("   Consider increasing n_estimators or max_depth")
    else:
        print("🚨 Poor performance (MAE > 1.2)")
        print("   Check for issues with featurization or context extraction")

    if r2 >= 0.75:
        print(f"✅ Good R² score ({r2:.3f} >= 0.75)")
    else:
        print(f"⚠️  Low R² score ({r2:.3f} < 0.75)")

    if corr > 0.3 and p_value < 0.05:
        print(f"✅ Uncertainty estimates are well-calibrated (r={corr:.3f})")
    else:
        print(f"⚠️  Uncertainty estimates may not be reliable (r={corr:.3f}, p={p_value:.4f})")

    # Save results
    results = {
        'mae': float(mae),
        'rmse': float(rmse),
        'r2': float(r2),
        'context_performance': context_performance,
        'uncertainty_correlation': float(corr),
        'uncertainty_pvalue': float(p_value),
        'n_train': len(train_df),
        'n_test': len(test_df),
        'trained_models': len(stats['trained_models']),
        'context_counts_train': {str(k): v for k, v in stats['counts'].items()}
    }

    with open('memory/esol_baseline_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Saved results to memory/esol_baseline_results.json")

    # Save predictions for later analysis
    pred_df = test_df.copy()
    pred_df['prediction'] = predictions
    pred_df['uncertainty'] = uncertainties
    pred_df['error'] = errors
    pred_df['context'] = [str(c) for c in contexts]

    pred_df.to_csv('memory/esol_baseline_predictions.csv', index=False)
    print(f"✅ Saved predictions to memory/esol_baseline_predictions.csv")

    # Save trained model
    model.save('memory/esol_baseline_model.pkl')
    print(f"✅ Saved model to memory/esol_baseline_model.pkl")

    print("\n" + "=" * 60)
    print("✅ Baseline experiment complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
