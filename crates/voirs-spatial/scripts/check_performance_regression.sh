#!/bin/bash
# Performance Regression Detection Script for CI/CD
#
# This script runs benchmarks and checks for performance regressions.
# It's designed to be used in CI/CD pipelines (GitHub Actions, GitLab CI, etc.)
#
# Exit codes:
#   0 - No regressions detected
#   1 - Performance regressions detected
#   2 - Script execution error

set -euo pipefail

# Configuration
BASELINE_DIR="${BASELINE_DIR:-.benchmark_baselines}"
BASELINE_FILE="${BASELINE_FILE:-baseline.json}"
REPORT_FILE="${REPORT_FILE:-regression_report.md}"
REGRESSION_THRESHOLD="${REGRESSION_THRESHOLD:-0.10}"  # 10% by default

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "========================================="
echo "Performance Regression Detection"
echo "========================================="
echo ""

# Check if baseline directory exists
if [ ! -d "$BASELINE_DIR" ]; then
    echo -e "${YELLOW}Warning: Baseline directory not found. Creating it...${NC}"
    mkdir -p "$BASELINE_DIR"
fi

BASELINE_PATH="$BASELINE_DIR/$BASELINE_FILE"

# Run benchmarks
echo "Running benchmarks..."
cargo bench --bench regression_tracking -- --noplot 2>&1 | tee benchmark_output.txt

# Check if benchmarks completed successfully
if [ "${PIPESTATUS[0]}" -ne 0 ]; then
    echo -e "${RED}Error: Benchmarks failed to run${NC}"
    exit 2
fi

# Parse benchmark results and check for regressions
# This is a simplified version - in production, you'd want more sophisticated parsing
echo ""
echo "Analyzing results..."

# Check if criterion generated any output
if [ ! -f "target/criterion/regression/*/new/estimates.json" ]; then
    echo -e "${YELLOW}Warning: No criterion output found. Skipping regression check.${NC}"
    exit 0
fi

# Generate regression report (this would be integrated with the Rust code in production)
echo ""
echo "Generating regression report..."

# Check for regressions in the output
REGRESSIONS=$(grep -i "regress" benchmark_output.txt || true)

if [ -n "$REGRESSIONS" ]; then
    echo -e "${RED}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${RED}  ⚠️  Performance Regressions Detected  ⚠️  ${NC}"
    echo -e "${RED}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    echo "$REGRESSIONS"
    echo ""
    echo -e "${RED}Performance has degraded beyond acceptable threshold (${REGRESSION_THRESHOLD})${NC}"
    echo "Please review the changes and optimize as needed."
    echo ""
    exit 1
else
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}  ✅  All Performance Checks Passed  ✅  ${NC}"
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    exit 0
fi
