#!/bin/bash
# Quick check of pilot results
# Usage: ./quick_check.sh [results_directory]

RESULTS_DIR=${1:-"results/ace_pilot_v2"}

echo "=================================================================="
echo "QUICK RESULTS CHECK: $RESULTS_DIR"
echo "=================================================================="
echo ""

if [ ! -d "$RESULTS_DIR/raw" ]; then
    echo "❌ Directory not found: $RESULTS_DIR/raw"
    exit 1
fi

# Count episodes
EPISODE_COUNT=$(ls $RESULTS_DIR/raw/*.json 2>/dev/null | wc -l | tr -d ' ')
echo "Episodes completed: $EPISODE_COUNT/40"
echo ""

# Agent distribution
echo "Agent distribution:"
python3 -c "
import json, glob, sys
from collections import Counter

pattern = sys.argv[1] + '/raw/*.json'
episodes = [json.load(open(f)) for f in glob.glob(pattern)]

if not episodes:
    print('  No episodes found')
    sys.exit(1)

agents = Counter(e['agent_type'] for e in episodes)
for agent, count in sorted(agents.items()):
    marker = '✅' if agent == 'a_c_e' else '  '
    print(f'{marker} {agent}: {count}')

# Check for ACE
if 'a_c_e' in agents:
    print()
    print('✅ ACE agent found!')
else:
    print()
    print('❌ ACE agent MISSING!')
" "$RESULTS_DIR"

echo ""

# Check failures
if [ -f "$RESULTS_DIR/failed_episodes.json" ]; then
    FAILED=$(python3 -c "import json, sys; print(len(json.load(open(sys.argv[1]))))" "$RESULTS_DIR/failed_episodes.json" 2>/dev/null || echo "0")
    if [ "$FAILED" -gt 0 ]; then
        echo "Failed episodes: $FAILED"
        echo ""
    fi
fi

echo "=================================================================="
