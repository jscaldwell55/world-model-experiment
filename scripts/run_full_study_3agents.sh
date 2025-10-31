#!/bin/bash
# run_full_study_3agents.sh
# Execute full study with 3 agents (Observer, Actor, ACE)
# 603 episodes: 3 environments × 3 agents × 67 seeds

set -euo pipefail  # Exit on error, undefined vars, pipe failures

# Configuration
CONFIG="config_full_study_3agents.yaml"
OUTPUT_DIR="results/full_study_3agents"
WORKERS=10
GIT_TAG="full-study-v1.1-3agents"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "========================================="
echo "Full Study Launch: 3 Agents (v1.1)"
echo "========================================="
echo ""

# 1. Check prerequisites
echo -e "${YELLOW}[1/6] Checking prerequisites...${NC}"

if [ ! -f "$CONFIG" ]; then
    echo -e "${RED}ERROR: Config file not found: $CONFIG${NC}"
    exit 1
fi

if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
    echo -e "${RED}ERROR: ANTHROPIC_API_KEY not set${NC}"
    exit 1
fi

if [ -z "${OPENAI_API_KEY:-}" ]; then
    echo -e "${RED}ERROR: OPENAI_API_KEY not set (needed for judge)${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Config file found${NC}"
echo -e "${GREEN}✓ API keys configured${NC}"

# 2. Check git status
echo ""
echo -e "${YELLOW}[2/6] Checking git status...${NC}"

if ! git diff-index --quiet HEAD --; then
    echo -e "${RED}ERROR: Uncommitted changes detected${NC}"
    echo "Please commit all changes before running full study."
    git status --short
    exit 1
fi

CURRENT_SHA=$(git rev-parse HEAD)
echo -e "${GREEN}✓ No uncommitted changes${NC}"
echo "Current SHA: $CURRENT_SHA"

# 3. Create git tag for provenance
echo ""
echo -e "${YELLOW}[3/6] Creating git tag...${NC}"

if git rev-parse "$GIT_TAG" >/dev/null 2>&1; then
    echo -e "${YELLOW}⚠ Tag $GIT_TAG already exists, skipping${NC}"
else
    git tag -a "$GIT_TAG" -m "Full study launch: 3 agents, 603 episodes (preregistration v1.1)"
    echo -e "${GREEN}✓ Created tag: $GIT_TAG${NC}"
fi

# 4. Confirm with user
echo ""
echo -e "${YELLOW}[4/6] Ready to launch full study${NC}"
echo ""
echo "Configuration:"
echo "  - Config: $CONFIG"
echo "  - Agents: observer, actor, a_c_e"
echo "  - Episodes: 603 (3 envs × 3 agents × 67 seeds)"
echo "  - Output: $OUTPUT_DIR"
echo "  - Workers: $WORKERS"
echo "  - Estimated cost: ~\$241"
echo "  - Estimated time: 3-4 hours"
echo ""
echo "Git provenance:"
echo "  - SHA: $CURRENT_SHA"
echo "  - Tag: $GIT_TAG"
echo ""

read -p "Proceed with full study? (yes/no): " -r
echo
if [[ ! $REPLY =~ ^[Yy]es$ ]]; then
    echo -e "${RED}Aborted by user${NC}"
    exit 1
fi

# 5. Run experiment
echo ""
echo -e "${YELLOW}[5/6] Launching experiment...${NC}"
echo "Start time: $(date)"
echo ""

python scripts/run_experiment_parallel.py \
    --config "$CONFIG" \
    --output-dir "$OUTPUT_DIR" \
    --workers "$WORKERS" \
    || {
        echo ""
        echo -e "${RED}ERROR: Experiment failed${NC}"
        exit 1
    }

echo ""
echo -e "${GREEN}✓ Experiment completed${NC}"
echo "End time: $(date)"

# 6. Generate analysis
echo ""
echo -e "${YELLOW}[6/6] Generating analysis...${NC}"

# Count completed episodes
COMPLETED=$(find "$OUTPUT_DIR/raw" -name "*.json" | wc -l | tr -d ' ')
EXPECTED=603

echo "Episodes completed: $COMPLETED / $EXPECTED"

if [ "$COMPLETED" -lt "$EXPECTED" ]; then
    echo -e "${YELLOW}⚠ Warning: Some episodes failed ($((EXPECTED - COMPLETED)) missing)${NC}"
fi

# Generate summary
if [ -f "scripts/generate_pilot_figures.py" ]; then
    echo "Generating Pareto plot and summary..."
    python scripts/generate_pilot_figures.py "$OUTPUT_DIR" || {
        echo -e "${YELLOW}⚠ Warning: Figure generation failed${NC}"
    }
fi

# Run statistical analysis if available
if [ -f "scripts/analyze_with_statistics.py" ]; then
    echo "Running statistical analysis..."
    python scripts/analyze_with_statistics.py "$OUTPUT_DIR" || {
        echo -e "${YELLOW}⚠ Warning: Statistical analysis failed${NC}"
    }
fi

echo ""
echo "========================================="
echo -e "${GREEN}FULL STUDY COMPLETE${NC}"
echo "========================================="
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Next steps:"
echo "  1. Review results: $OUTPUT_DIR/summary.json"
echo "  2. Check Pareto plot: $OUTPUT_DIR/pareto_plot.png"
echo "  3. Verify H1a (accuracy ≥70%) and H1b (cost ≤50% Actor)"
echo "  4. Run statistical tests: scripts/analyze_with_statistics.py"
echo "  5. Update CHANGELOG.md with any post-hoc discoveries"
echo ""
echo "Git tag for reproducibility: $GIT_TAG"
echo "SHA: $CURRENT_SHA"
echo ""
