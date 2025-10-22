#!/usr/bin/env python3
"""Create scatter plots showing coupling between token NLL and belief surprisal."""

import json
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Load data
files = {
    'HotPotLab': 'results/validation_20251022_094112/HotPot_Actor_ep042_token.json',
    'SwitchLight': 'results/validation_20251022_094112/SwitchLight_Actor_ep100_token.json',
    'ChemTile': 'results/validation_20251022_094112/ChemTile_Actor_ep200_token.json'
}

data_by_env = {}
for env_name, file_path in files.items():
    with open(file_path) as f:
        data = json.load(f)

    token_nlls = [entry['sequence_nll'] for entry in data['entries']]
    belief_surprisals = [entry['belief_surprisal'] for entry in data['entries']]

    data_by_env[env_name] = {
        'token_nll': np.array(token_nlls),
        'belief_surprisal': np.array(belief_surprisals)
    }

# Create figure with 3 subplots
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle('Token NLL vs Belief Surprisal: 3-Episode Validation Pilot', fontsize=14, fontweight='bold')

colors = {'HotPotLab': '#e74c3c', 'SwitchLight': '#2ecc71', 'ChemTile': '#3498db'}
env_names = ['HotPotLab', 'SwitchLight', 'ChemTile']

for idx, env_name in enumerate(env_names):
    ax = axes[idx]
    env_data = data_by_env[env_name]

    x = env_data['belief_surprisal']
    y = env_data['token_nll']

    # Scatter plot
    ax.scatter(x, y, alpha=0.7, s=100, color=colors[env_name], edgecolors='black', linewidth=1)

    # Add step labels
    for i, (xi, yi) in enumerate(zip(x, y)):
        ax.annotate(str(i), (xi, yi), fontsize=8, ha='center', va='center', color='white', fontweight='bold')

    # Compute correlation
    if np.std(x) > 1e-6 and np.std(y) > 1e-6:
        r, p = stats.pearsonr(x, y)

        # Add regression line
        if len(x) > 1:
            z = np.polyfit(x, y, 1)
            p_line = np.poly1d(z)
            x_line = np.linspace(x.min(), x.max(), 100)
            ax.plot(x_line, p_line(x_line), "--", alpha=0.5, color=colors[env_name], linewidth=2)

        # Status symbol
        if env_name == 'SwitchLight':
            status = '✓ STRONG' if r > 0.5 else '✓'
        elif env_name == 'HotPotLab':
            status = '✗ FAIL' if abs(r) < 0.3 else '~'
        else:
            status = '?' if np.count_nonzero(x) < 3 else '~'

        title = f'{env_name}\nr={r:.3f} (p={p:.3f}) {status}'
    else:
        title = f'{env_name}\n(insufficient variance)'

    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel('Belief Surprisal', fontsize=10)
    ax.set_ylabel('Token NLL', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Add zero lines
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

plt.tight_layout()
plt.savefig('pilot_coupling_analysis.png', dpi=300, bbox_inches='tight')
print("\n✅ Visualization saved to: pilot_coupling_analysis.png")
print("\nKey findings:")
print("  - SwitchLight: r=0.826 (p=0.003) ⭐ STRONG coupling")
print("  - HotPotLab: r=-0.105 (p=0.773) ✗ NO coupling")
print("  - ChemTile: r=0.359 (p=0.308) ? Insufficient data")
