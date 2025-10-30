#!/bin/bash
# Monitored pilot run with real-time progress tracking
# Runs the corrected ACE pilot and monitors for issues

set -e

echo "=================================================================="
echo "MONITORED ACE PILOT RUN"
echo "=================================================================="
echo ""

# ============================================================
# 1. Pre-flight check
# ============================================================
echo "Step 1: Running pre-flight check..."
if ! ./pre_flight_check.sh; then
    echo "❌ Pre-flight check failed. Aborting."
    exit 1
fi
echo ""

# ============================================================
# 2. Backup old results if they exist
# ============================================================
echo "Step 2: Backing up old results (if any)..."
if [ -d "results/ace_pilot_v2" ]; then
    BACKUP_DIR="results/ace_pilot_v2_backup_$(date +%Y%m%d_%H%M%S)"
    mv results/ace_pilot_v2 "$BACKUP_DIR"
    echo "   Backed up to: $BACKUP_DIR"
fi
mkdir -p results/ace_pilot_v2/raw
echo "   ✅ Output directory ready"
echo ""

# ============================================================
# 3. Start pilot
# ============================================================
echo "Step 3: Starting pilot experiment..."
echo "   Config: config_ace_pilot_v2.yaml"
echo "   Output: results/ace_pilot_v2"
echo "   Workers: 6"
echo "   Start time: $(date)"
echo ""

# Run in background and capture PID
python scripts/run_experiment_parallel.py \
    --config config_ace_pilot_v2.yaml \
    --preregistration preregistration.yaml \
    --output-dir results/ace_pilot_v2 \
    --workers 6 &

PILOT_PID=$!
echo "   Pilot PID: $PILOT_PID"
echo ""

# ============================================================
# 4. Monitor progress
# ============================================================
echo "Step 4: Monitoring progress (Ctrl+C to view final state)..."
echo "=================================================================="
echo ""

LAST_COUNT=0
START_TIME=$(date +%s)

while kill -0 $PILOT_PID 2>/dev/null; do
    sleep 30

    # Count episodes
    EPISODES=$(ls results/ace_pilot_v2/raw/*.json 2>/dev/null | wc -l | tr -d ' ')
    ELAPSED=$(($(date +%s) - START_TIME))
    ELAPSED_MIN=$((ELAPSED / 60))

    # Show progress if count changed
    if [ "$EPISODES" != "$LAST_COUNT" ]; then
        echo "[$(date +%H:%M:%S)] Episodes completed: $EPISODES/40 (elapsed: ${ELAPSED_MIN}m)"
        LAST_COUNT=$EPISODES

        # Check agent distribution every 10 episodes
        if [ $((EPISODES % 10)) -eq 0 ] && [ $EPISODES -gt 0 ]; then
            echo "  Agent distribution so far:"
            python3 -c "
import json, glob
from collections import Counter
episodes = [json.load(open(f)) for f in glob.glob('results/ace_pilot_v2/raw/*.json')]
agents = Counter(e['agent_type'] for e in episodes)
for agent, count in sorted(agents.items()):
    indicator = '✅' if agent == 'a_c_e' else '  '
    print(f'    {indicator} {agent}: {count}')
            "
            echo ""
        fi
    fi

    # Check for errors
    if [ -f results/ace_pilot_v2/failed_episodes.json ]; then
        FAILED=$(python3 -c "import json; print(len(json.load(open('results/ace_pilot_v2/failed_episodes.json'))))" 2>/dev/null || echo "0")
        if [ "$FAILED" -gt 0 ]; then
            echo "  ⚠️  Failed episodes: $FAILED"
        fi
    fi
done

# Wait for final completion
wait $PILOT_PID
PILOT_EXIT_CODE=$?

# ============================================================
# 5. Final verification
# ============================================================
echo ""
echo "=================================================================="
echo "PILOT COMPLETED"
echo "=================================================================="
echo ""

FINAL_COUNT=$(ls results/ace_pilot_v2/raw/*.json 2>/dev/null | wc -l | tr -d ' ')
TOTAL_TIME=$(($(date +%s) - START_TIME))
TOTAL_MIN=$((TOTAL_TIME / 60))

echo "Final results:"
echo "  Episodes completed: $FINAL_COUNT/40"
echo "  Total time: ${TOTAL_MIN}m"
echo "  End time: $(date)"
echo ""

# Check agent distribution
echo "Agent distribution:"
python3 -c "
import json, glob
from collections import Counter

episodes = [json.load(open(f)) for f in glob.glob('results/ace_pilot_v2/raw/*.json')]
agents = Counter(e['agent_type'] for e in episodes)

expected = {'observer': 10, 'actor': 10, 'model_based': 10, 'a_c_e': 10}
all_good = True

for agent_type, count in sorted(agents.items()):
    expected_count = expected.get(agent_type, 0)
    if count == expected_count:
        print(f'  ✅ {agent_type}: {count}/{expected_count}')
    else:
        print(f'  ❌ {agent_type}: {count}/{expected_count}')
        all_good = False

# Check for missing agents
for agent_type, expected_count in expected.items():
    if agent_type not in agents:
        print(f'  ❌ {agent_type}: 0/{expected_count} (MISSING)')
        all_good = False

if all_good and len(episodes) == 40:
    print()
    print('✅ ALL AGENTS RAN SUCCESSFULLY!')
    print('✅ ACE (a_c_e) is present!')
    print('✅ Ready for hypothesis testing!')
"

echo ""

# Check for failures
if [ -f results/ace_pilot_v2/failed_episodes.json ]; then
    FAILED=$(python3 -c "import json; print(len(json.load(open('results/ace_pilot_v2/failed_episodes.json'))))" 2>/dev/null || echo "0")
    if [ "$FAILED" -gt 0 ]; then
        echo "⚠️  $FAILED episodes failed (see results/ace_pilot_v2/failed_episodes.json)"
        echo ""

        # Show error types
        echo "Failed episode types:"
        python3 -c "
import json
from collections import Counter

failed = json.load(open('results/ace_pilot_v2/failed_episodes.json'))
errors = []
for f in failed:
    data = json.loads(f)
    errors.append(data.get('error_type', 'Unknown'))

error_counts = Counter(errors)
for error_type, count in error_counts.items():
    print(f'  - {error_type}: {count}')
        "
        echo ""
    fi
fi

echo "=================================================================="
if [ $PILOT_EXIT_CODE -eq 0 ] && [ $FINAL_COUNT -ge 38 ]; then
    echo "✅ PILOT SUCCESSFUL!"
else
    echo "⚠️  PILOT COMPLETED WITH ISSUES"
fi
echo "=================================================================="

exit $PILOT_EXIT_CODE
