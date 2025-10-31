#!/usr/bin/env python3
"""Fix the run_experiment_parallel.py to read agents from config"""

import sys

# Read the file
with open('scripts/run_experiment_parallel.py', 'r') as f:
    lines = f.readlines()

# Find and fix line 199 (index 198)
for i, line in enumerate(lines):
    if i == 198 and 'for agent_name, agent_cls in agent_mapping.items():' in line:
        # Get the indentation
        indent = len(line) - len(line.lstrip())
        spaces = ' ' * indent
        
        # Replace with fixed version
        lines[i] = f'{spaces}for agent_name in config.get("agents", agent_mapping.keys()):\n'
        lines.insert(i + 1, f'{spaces}    agent_cls = agent_mapping[agent_name]\n')
        
        print("✅ Fixed line 199")
        print(f"   Old: {line.strip()}")
        print(f"   New: for agent_name in config.get('agents', agent_mapping.keys()):")
        print(f"        + added: agent_cls = agent_mapping[agent_name]")
        break
else:
    print("❌ Could not find the line to fix")
    sys.exit(1)

# Back up original
with open('scripts/run_experiment_parallel.py.backup', 'w') as f:
    # Read original again for backup
    with open('scripts/run_experiment_parallel.py', 'r') as orig:
        f.write(orig.read())

# Write fixed version
with open('scripts/run_experiment_parallel.py', 'w') as f:
    f.writelines(lines)

print("✅ File fixed and backed up")
print("\nVerifying fix:")

# Show the fixed lines
with open('scripts/run_experiment_parallel.py', 'r') as f:
    fixed_lines = f.readlines()
    print("Lines 198-202:")
    for i in range(197, 202):
        print(f"  {i+1}: {fixed_lines[i].rstrip()}")
