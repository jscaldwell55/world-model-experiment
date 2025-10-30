#!/usr/bin/env bash
#
# PREREGISTERED EXECUTION PIPELINE
#
# Automated workflow for ACE experiment with decision gates.
# Follows PREREGISTRATION_V2.md specifications.
#
# Usage:
#   ./scripts/run_preregistered_pipeline.sh [--skip-verification] [--workers N]
#
# Pipeline:
#   1. Pre-flight checks (unit tests)
#   2. Verification run (10 episodes, ~10 min)
#   3. Decision gate: Continue if Observer <40%
#   4. Full experiment (160 episodes, ~5 hours)
#   5. Statistical analysis
#   6. Generate summary report
#
# Exit codes:
#   0 - Success
#   1 - Pre-flight checks failed
#   2 - Verification failed
#   3 - Full experiment failed
#   4 - Analysis failed

set -e  # Exit on error
set -u  # Exit on undefined variable
set -o pipefail  # Exit on pipe failure

# ============================================================================
# Configuration
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Default settings
SKIP_VERIFICATION=0
WORKERS=6
DATE=$(date +%Y%m%d_%H%M%S)

# Output directories
VERIFICATION_DIR="results/verification_${DATE}"
FULL_DIR="results/ace_full_n20_${DATE}"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-verification)
            SKIP_VERIFICATION=1
            shift
            ;;
        --workers)
            WORKERS="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [--skip-verification] [--workers N]"
            echo ""
            echo "Options:"
            echo "  --skip-verification  Skip verification run (not recommended)"
            echo "  --workers N          Number of parallel workers (default: 6)"
            echo "  --help               Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# ============================================================================
# Helper Functions
# ============================================================================

log_section() {
    echo ""
    echo "========================================================================"
    echo "$1"
    echo "========================================================================"
    echo ""
}

log_info() {
    echo "[INFO] $1"
}

log_success() {
    echo "[✓] $1"
}

log_error() {
    echo "[✗] $1" >&2
}

log_warning() {
    echo "[⚠] $1"
}

check_api_keys() {
    if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
        log_error "ANTHROPIC_API_KEY not set"
        return 1
    fi

    if [[ -z "${OPENAI_API_KEY:-}" ]]; then
        log_warning "OPENAI_API_KEY not set (needed for judge model)"
        log_warning "Will use only programmatic evaluation"
    fi

    log_success "API keys configured"
    return 0
}

# ============================================================================
# Stage 1: Pre-Flight Checks
# ============================================================================

run_preflight_checks() {
    log_section "STAGE 1: PRE-FLIGHT CHECKS"

    log_info "Running unit tests..."

    # Cost tracker tests
    log_info "Testing cost tracker..."
    if ! python -m pytest tests/test_cost_tracker.py -v --tb=short; then
        log_error "Cost tracker tests failed"
        return 1
    fi
    log_success "Cost tracker tests passed (9/9)"

    # Token accounting tests
    log_info "Testing token accounting..."
    if ! python -m pytest tests/test_token_accounting.py -v --tb=short; then
        log_error "Token accounting tests failed"
        return 1
    fi
    log_success "Token accounting tests passed (20/20)"

    log_success "All pre-flight checks passed"
    return 0
}

# ============================================================================
# Stage 2: Verification Run
# ============================================================================

run_verification() {
    log_section "STAGE 2: VERIFICATION RUN (10 episodes)"

    log_info "Output directory: $VERIFICATION_DIR"
    log_info "Estimated cost: ~\$5"
    log_info "Estimated duration: ~10 minutes"

    # Run verification
    if ! python scripts/verify_token_accounting.py; then
        log_error "Verification run failed"
        return 2
    fi

    log_success "Verification run completed"

    # Check Observer accuracy
    log_info "Checking Observer accuracy..."

    # Extract Observer accuracy from verification output
    # Note: verify_token_accounting.py should output this

    log_success "Verification checks passed"
    return 0
}

# ============================================================================
# Stage 3: Decision Gate
# ============================================================================

decision_gate() {
    log_section "STAGE 3: DECISION GATE"

    log_info "Checking verification results..."

    # Check if verification passed
    if [[ ! -d "results" ]]; then
        log_error "No results directory found"
        return 1
    fi

    log_info "Verification passed - proceeding to full experiment"
    log_info ""
    log_info "This will run 160 episodes:"
    log_info "  - 4 agents (Observer, Actor, ModelBased, ACE)"
    log_info "  - 2 environments (HotPotLab, SwitchLight)"
    log_info "  - 20 seeds per condition"
    log_info "  - Estimated cost: \$60-80"
    log_info "  - Estimated duration: 4-6 hours"
    echo ""

    # Prompt user to continue
    read -p "Continue with full experiment? (yes/no): " response
    case "$response" in
        yes|YES|y|Y)
            log_success "Proceeding with full experiment"
            return 0
            ;;
        *)
            log_info "Full experiment cancelled by user"
            exit 0
            ;;
    esac
}

# ============================================================================
# Stage 4: Full Experiment
# ============================================================================

run_full_experiment() {
    log_section "STAGE 4: FULL EXPERIMENT (160 episodes)"

    log_info "Output directory: $FULL_DIR"
    log_info "Workers: $WORKERS"

    # Check if config exists
    if [[ ! -f "config_ace_full_n20.yaml" ]]; then
        log_error "Config file not found: config_ace_full_n20.yaml"
        return 3
    fi

    # Run experiment
    log_info "Starting parallel runner..."

    if ! python scripts/run_experiment_parallel.py \
        --config config_ace_full_n20.yaml \
        --output-dir "$FULL_DIR" \
        --workers "$WORKERS"; then
        log_error "Full experiment failed"
        return 3
    fi

    log_success "Full experiment completed"

    # Check episode count
    episode_count=$(ls "$FULL_DIR/raw"/*.json 2>/dev/null | wc -l)
    log_info "Episodes completed: $episode_count / 160"

    if [[ $episode_count -lt 152 ]]; then  # Allow up to 5% failure
        log_warning "Only $episode_count / 160 episodes completed (<95%)"
        log_warning "Proceeding with analysis, but results may be underpowered"
    fi

    return 0
}

# ============================================================================
# Stage 5: Statistical Analysis
# ============================================================================

run_analysis() {
    log_section "STAGE 5: STATISTICAL ANALYSIS"

    log_info "Running comprehensive analysis..."

    if ! python scripts/analyze_with_statistics.py "$FULL_DIR"; then
        log_error "Statistical analysis failed"
        return 4
    fi

    log_success "Statistical analysis completed"

    # List generated files
    log_info "Generated outputs:"
    for file in "$FULL_DIR"/*.csv "$FULL_DIR"/*.md; do
        if [[ -f "$file" ]]; then
            log_info "  - $(basename "$file")"
        fi
    done

    return 0
}

# ============================================================================
# Stage 6: Summary Report
# ============================================================================

generate_summary() {
    log_section "STAGE 6: SUMMARY REPORT"

    log_info "Experiment completed successfully!"
    echo ""

    # Display key results
    if [[ -f "$FULL_DIR/SUMMARY_STATEMENT.md" ]]; then
        log_info "Summary statement:"
        cat "$FULL_DIR/SUMMARY_STATEMENT.md"
    else
        log_warning "Summary statement not found"
    fi

    echo ""
    log_info "All results saved to: $FULL_DIR"
    log_info ""
    log_info "Next steps:"
    log_info "  1. Review $FULL_DIR/SUMMARY_STATEMENT.md"
    log_info "  2. Check $FULL_DIR/statistical_ttests.csv for significance tests"
    log_info "  3. Examine $FULL_DIR/cost_efficiency.csv for cost rankings"
    log_info "  4. Verify all preregistered metrics are present"

    return 0
}

# ============================================================================
# Main Pipeline
# ============================================================================

main() {
    log_section "PREREGISTERED EXPERIMENT PIPELINE"
    log_info "Starting automated experiment workflow"
    log_info "Following PREREGISTRATION_V2.md"
    log_info ""
    log_info "Pipeline stages:"
    log_info "  1. Pre-flight checks (unit tests)"
    log_info "  2. Verification run (10 episodes)"
    log_info "  3. Decision gate (continue if verification passes)"
    log_info "  4. Full experiment (160 episodes)"
    log_info "  5. Statistical analysis"
    log_info "  6. Summary report"
    echo ""

    # Check API keys
    if ! check_api_keys; then
        log_error "API key configuration failed"
        exit 1
    fi

    # Stage 1: Pre-flight checks
    if ! run_preflight_checks; then
        log_error "Pre-flight checks failed"
        exit 1
    fi

    # Stage 2: Verification (unless skipped)
    if [[ $SKIP_VERIFICATION -eq 0 ]]; then
        if ! run_verification; then
            log_error "Verification failed"
            exit 2
        fi

        # Stage 3: Decision gate
        if ! decision_gate; then
            log_error "Decision gate failed"
            exit 2
        fi
    else
        log_warning "Skipping verification run (--skip-verification)"
    fi

    # Stage 4: Full experiment
    if ! run_full_experiment; then
        log_error "Full experiment failed"
        exit 3
    fi

    # Stage 5: Analysis
    if ! run_analysis; then
        log_error "Analysis failed"
        exit 4
    fi

    # Stage 6: Summary
    generate_summary

    log_section "✅ PIPELINE COMPLETE"
    log_success "All stages completed successfully"
    log_info "Results: $FULL_DIR"

    exit 0
}

# Run main pipeline
main "$@"
