#!/bin/bash
# Pre-flight check for ACE pilot experiment
# Validates everything is ready before running the full 40-episode pilot

set -e  # Exit on any error

echo "=================================================================="
echo "ACE PILOT PRE-FLIGHT CHECK"
echo "=================================================================="
echo ""

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Track overall status
ALL_PASSED=true

# ============================================================
# 1. Check API Keys
# ============================================================
echo "1. Checking API keys..."
if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo -e "   ${RED}❌ ANTHROPIC_API_KEY not set${NC}"
    echo "      Run: export ANTHROPIC_API_KEY='your-key'"
    ALL_PASSED=false
else
    echo -e "   ${GREEN}✅ ANTHROPIC_API_KEY set${NC}"
fi

if [ -z "$OPENAI_API_KEY" ]; then
    echo -e "   ${YELLOW}⚠️  OPENAI_API_KEY not set (needed for judge)${NC}"
    echo "      Run: export OPENAI_API_KEY='your-key'"
    ALL_PASSED=false
else
    echo -e "   ${GREEN}✅ OPENAI_API_KEY set${NC}"
fi
echo ""

# ============================================================
# 2. Check Configuration File
# ============================================================
echo "2. Checking configuration file..."
if [ ! -f "config_ace_pilot_v2.yaml" ]; then
    echo -e "   ${RED}❌ config_ace_pilot_v2.yaml not found${NC}"
    ALL_PASSED=false
else
    echo -e "   ${GREEN}✅ config_ace_pilot_v2.yaml exists${NC}"

    # Validate YAML syntax
    if python3 -c "import yaml; yaml.safe_load(open('config_ace_pilot_v2.yaml'))" 2>/dev/null; then
        echo -e "   ${GREEN}✅ YAML syntax valid${NC}"
    else
        echo -e "   ${RED}❌ YAML syntax invalid${NC}"
        ALL_PASSED=false
    fi

    # Check for critical fix: 'a_c_e' not 'ace'
    if grep -q "^  a_c_e:" config_ace_pilot_v2.yaml; then
        echo -e "   ${GREEN}✅ Contains 'a_c_e' model config (FIX APPLIED)${NC}"
    else
        echo -e "   ${RED}❌ Missing 'a_c_e' model config${NC}"
        echo "      Config should have 'a_c_e:' not 'ace:'"
        ALL_PASSED=false
    fi
fi
echo ""

# ============================================================
# 3. Check Agent Imports
# ============================================================
echo "3. Checking agent imports..."
AGENTS=("ObserverAgent" "ActorAgent" "ModelBasedAgent" "ACEAgent")
for agent in "${AGENTS[@]}"; do
    if python3 -c "from agents.${agent,,} import ${agent}" 2>/dev/null; then
        echo -e "   ${GREEN}✅ ${agent} can be imported${NC}"
    else
        echo -e "   ${RED}❌ ${agent} import failed${NC}"
        ALL_PASSED=false
    fi
done
echo ""

# ============================================================
# 4. Check Agent Type Conversion
# ============================================================
echo "4. Checking agent type conversion..."
AGENT_TYPE=$(python3 -c "
import re
class_name = 'ACE'  # ACEAgent with 'Agent' removed
agent_type = re.sub(r'(?<!^)(?=[A-Z])', '_', class_name).lower()
print(agent_type)
")

if [ "$AGENT_TYPE" == "a_c_e" ]; then
    echo -e "   ${GREEN}✅ ACEAgent converts to 'a_c_e'${NC}"
else
    echo -e "   ${RED}❌ ACEAgent converts to '$AGENT_TYPE' (expected 'a_c_e')${NC}"
    ALL_PASSED=false
fi
echo ""

# ============================================================
# 5. Check Environment Imports
# ============================================================
echo "5. Checking environment imports..."
ENVS=("HotPotLab" "SwitchLight")
for env in "${ENVS[@]}"; do
    if python3 -c "from environments.${env,,//_/} import ${env}" 2>/dev/null; then
        echo -e "   ${GREEN}✅ ${env} can be imported${NC}"
    else
        echo -e "   ${RED}❌ ${env} import failed${NC}"
        ALL_PASSED=false
    fi
done
echo ""

# ============================================================
# 6. Check Output Directory
# ============================================================
echo "6. Checking output directory..."
if [ -d "results/ace_pilot_v2" ]; then
    FILE_COUNT=$(ls results/ace_pilot_v2/raw/*.json 2>/dev/null | wc -l | tr -d ' ')
    if [ "$FILE_COUNT" -gt 0 ]; then
        echo -e "   ${YELLOW}⚠️  results/ace_pilot_v2 already exists with $FILE_COUNT episodes${NC}"
        echo "      Consider backing up or removing before rerun"
    else
        echo -e "   ${GREEN}✅ results/ace_pilot_v2 exists (empty)${NC}"
    fi
else
    mkdir -p results/ace_pilot_v2/raw
    echo -e "   ${GREEN}✅ Created results/ace_pilot_v2${NC}"
fi
echo ""

# ============================================================
# 7. Optional: Test ACE Single Episode
# ============================================================
if [ -f "test_ace_single.py" ]; then
    echo "7. OPTIONAL: Run ACE single episode test?"
    echo "   This will make real API calls (~$0.10)"
    echo "   Skip this if you're confident (y/N): "
    read -t 10 -n 1 RUN_TEST || RUN_TEST="n"
    echo ""

    if [ "$RUN_TEST" == "y" ] || [ "$RUN_TEST" == "Y" ]; then
        echo "   Running test_ace_single.py..."
        if python3 test_ace_single.py; then
            echo -e "   ${GREEN}✅ ACE single episode test PASSED${NC}"
        else
            echo -e "   ${RED}❌ ACE single episode test FAILED${NC}"
            ALL_PASSED=false
        fi
    else
        echo "   ⏭️  Skipped single episode test"
    fi
    echo ""
fi

# ============================================================
# 8. Check Script Exists
# ============================================================
echo "8. Checking pilot script..."
if [ ! -f "reproduce.sh" ]; then
    echo -e "   ${RED}❌ reproduce.sh not found${NC}"
    ALL_PASSED=false
else
    echo -e "   ${GREEN}✅ reproduce.sh exists${NC}"

    # Make executable
    chmod +x reproduce.sh
    echo -e "   ${GREEN}✅ reproduce.sh is executable${NC}"
fi
echo ""

# ============================================================
# Final Summary
# ============================================================
echo "=================================================================="
echo "PRE-FLIGHT CHECK SUMMARY"
echo "=================================================================="

if [ "$ALL_PASSED" = true ]; then
    echo -e "${GREEN}✅ ALL CHECKS PASSED - Ready to run pilot!${NC}"
    echo ""
    echo "To run the corrected pilot:"
    echo "  ./reproduce.sh --config config_ace_pilot_v2.yaml"
    echo ""
    echo "Expected results:"
    echo "  - 40 episodes total"
    echo "  - 10 episodes per agent: observer, actor, model_based, a_c_e"
    echo "  - ~15-20 minutes runtime"
    echo "  - ~\$15-25 cost"
    echo ""
    exit 0
else
    echo -e "${RED}❌ SOME CHECKS FAILED - Fix issues before running pilot${NC}"
    echo ""
    echo "Review the errors above and fix before proceeding."
    echo ""
    exit 1
fi
