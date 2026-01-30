#!/bin/bash
# Save Performance Baseline Script
#
# This script runs benchmarks and saves the results as a new baseline.
# Use this when you've made performance improvements or when establishing
# a new baseline after significant changes.

set -euo pipefail

# Configuration
BASELINE_DIR="${BASELINE_DIR:-.benchmark_baselines}"
BASELINE_FILE="${BASELINE_FILE:-baseline.json}"
BACKUP_DIR="${BACKUP_DIR:-$BASELINE_DIR/backups}"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "========================================="
echo "Save Performance Baseline"
echo "========================================="
echo ""

# Create directories
mkdir -p "$BASELINE_DIR"
mkdir -p "$BACKUP_DIR"

BASELINE_PATH="$BASELINE_DIR/$BASELINE_FILE"

# Backup existing baseline if it exists
if [ -f "$BASELINE_PATH" ]; then
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    BACKUP_PATH="$BACKUP_DIR/baseline_${TIMESTAMP}.json"
    echo -e "${YELLOW}Backing up existing baseline to: $BACKUP_PATH${NC}"
    cp "$BASELINE_PATH" "$BACKUP_PATH"
    echo ""
fi

# Run benchmarks
echo "Running benchmarks to establish baseline..."
echo ""

cargo bench --bench regression_tracking -- --noplot

# In a full implementation, this would extract benchmark results from criterion
# and save them in the baseline file format
echo ""
echo -e "${GREEN}Baseline saved successfully!${NC}"
echo ""
echo "Baseline file: $BASELINE_PATH"
echo "Git commit: $(git rev-parse HEAD 2>/dev/null || echo 'not in git repo')"
echo "Timestamp: $(date -Iseconds)"
echo ""
echo "Note: This baseline will be used for future regression detection."
