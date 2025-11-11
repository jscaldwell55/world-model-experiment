#!/bin/bash
# reproduce.sh
# One-command pilot experiment runner (≤30 minutes)
#
# Usage:
#   export ANTHROPIC_API_KEY="your-key-here"
#   ./reproduce.sh

set -e  # Exit on any error

echo "=================================================="
echo "ACE Cost-Aware Pilot Experiment"
echo "=================================================="
echo ""

# ============================================================
# 1. Verify preregistration exists
# ============================================================
echo "Step 1: Verifying preregistration..."
if [ ! -f preregistration.md ]; then
    echo "❌ ERROR: preregistration.md not found"
    echo "   Experiments cannot run without preregistration"
    exit 1
fi

# Check git tag exists
if ! git tag -l | grep -q "prereg-v1.0"; then
    echo "⚠️  WARNING: prereg-v1.0 tag not found"
fi

echo "✅ Preregistration verified"
echo ""

# ============================================================
# 2. Check API key
# ============================================================
echo "Step 2: Checking API key..."
if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo "❌ ERROR: ANTHROPIC_API_KEY environment variable not set"
    echo "   Please run: export ANTHROPIC_API_KEY=\"your-key-here\""
    exit 1
fi

echo "✅ API key found"
echo ""

# ============================================================
# 3. Record provenance
# ============================================================
echo "Step 3: Recording provenance..."
GIT_SHA=$(git rev-parse HEAD)
TIMESTAMP=$(date -u +%Y-%m-%dT%H:%M:%SZ)

echo "   Git SHA: $GIT_SHA"
echo "   Timestamp: $TIMESTAMP"
echo "   Config: config_ace_pilot.yaml"
echo "   Preregistration: preregistration.md"
echo ""

# ============================================================
# 4. Run pilot experiment (40 episodes)
# ============================================================
echo "Step 4: Running pilot experiment (40 episodes)..."
echo "   Environments: HotPot, SwitchLight"
echo "   Agents: Observer, Actor, Model-Based, ACE"
echo "   Seeds: 5 per env-agent pair"
echo "   Workers: 6 parallel"
echo "   Estimated time: 15-20 minutes"
echo "   Estimated cost: \$15-25"
echo ""

START_TIME=$(date +%s)

# Run the experiment
python scripts/run_experiment_parallel.py \
    --config config_ace_pilot.yaml \
    --preregistration preregistration.yaml \
    --output-dir results/ace_pilot \
    --workers 6

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
MINUTES=$((DURATION / 60))
SECONDS=$((DURATION % 60))

echo ""
echo "✅ Pilot experiment completed"
echo "   Duration: ${MINUTES}m ${SECONDS}s (target: ≤30m)"
echo ""

# ============================================================
# 5. Verify outputs
# ============================================================
echo "Step 5: Verifying outputs..."

# Count episode logs
EPISODE_COUNT=$(ls results/ace_pilot/raw/*.json 2>/dev/null | wc -l | tr -d ' ')
echo "   Episode logs: $EPISODE_COUNT (expected: 40)"

if [ "$EPISODE_COUNT" -lt 40 ]; then
    echo "⚠️  WARNING: Found fewer than 40 episode logs"
fi

# Check for failed episodes
if [ -f results/ace_pilot/failed_episodes.json ]; then
    FAILED_COUNT=$(cat results/ace_pilot/failed_episodes.json | grep -c '"episode_id"' || echo "0")
    if [ "$FAILED_COUNT" -gt 0 ]; then
        echo "⚠️  WARNING: $FAILED_COUNT episodes failed"
        echo "   See results/ace_pilot/failed_episodes.json for details"
    fi
fi

echo ""

# ============================================================
# 6. Generate summary
# ============================================================
echo "Step 6: Generating summary..."

if [ -f analyze_ace_pilot.py ]; then
    echo "   Running analysis script..."
    python analyze_ace_pilot.py results/ace_pilot
    echo "✅ Analysis complete"
else
    echo "⚠️  WARNING: analyze_ace_pilot.py not found"
    echo "   You can analyze results manually from results/ace_pilot/raw/"
fi

echo ""

# ============================================================
# 7. Summary
# ============================================================
echo "=================================================="
echo "Pilot Experiment Summary"
echo "=================================================="
echo ""
echo "Results saved to: results/ace_pilot/"
echo ""
echo "Key files:"
echo "  - results/ace_pilot/raw/*.json          # Individual episode logs"
echo "  - results/ace_pilot/aggregate_metrics.csv  # Summary metrics (if analysis ran)"
echo "  - results/ace_pilot/summary.json        # Overall statistics (if analysis ran)"
echo ""
echo "Next steps:"
echo "  1. Review results in results/ace_pilot/"
echo "  2. Check if ACE is on Pareto frontier"
echo "  3. Decide whether to proceed to full experiment (600 episodes)"
echo ""
echo "To analyze results manually:"
echo "  python analyze_ace_pilot.py results/ace_pilot"
echo ""
echo "=================================================="
echo "Pilot complete! ✨"
echo "=================================================="
