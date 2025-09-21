#!/bin/bash

# Security vulnerability scanning automation script
# This script performs comprehensive security checks on the VoiRS Recognizer codebase

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
WORKSPACE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../../.."
REPORTS_DIR="${WORKSPACE_ROOT}/security-reports"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Ensure required tools are available
check_dependencies() {
    echo -e "${GREEN}Checking dependencies...${NC}"
    
    # Check cargo-audit
    if ! command -v cargo-audit &> /dev/null; then
        echo -e "${YELLOW}Installing cargo-audit...${NC}"
        cargo install cargo-audit
    fi
    
    # Check cargo-deny
    if ! command -v cargo-deny &> /dev/null; then
        echo -e "${YELLOW}Installing cargo-deny...${NC}"
        cargo install cargo-deny
    fi
    
    echo -e "${GREEN}Dependencies ready${NC}"
}

# Create reports directory
setup_reports() {
    mkdir -p "${REPORTS_DIR}"
    echo -e "${GREEN}Reports directory: ${REPORTS_DIR}${NC}"
}

# Run cargo audit for vulnerability scanning
run_security_audit() {
    echo -e "${GREEN}Running security vulnerability scan...${NC}"
    
    cd "${WORKSPACE_ROOT}"
    
    # Run with JSON output for processing
    if cargo audit --json > "${REPORTS_DIR}/security-audit-${TIMESTAMP}.json"; then
        echo -e "${GREEN}✓ No vulnerabilities found${NC}"
        return 0
    else
        echo -e "${RED}✗ Vulnerabilities detected${NC}"
        # Also generate human-readable report
        cargo audit > "${REPORTS_DIR}/security-audit-${TIMESTAMP}.txt" 2>&1 || true
        return 1
    fi
}

# Run cargo deny for additional security checks
run_cargo_deny() {
    echo -e "${GREEN}Running cargo-deny checks...${NC}"
    
    cd "${WORKSPACE_ROOT}"
    
    # Check for deny.toml existence
    if [ ! -f "deny.toml" ]; then
        echo -e "${YELLOW}Creating deny.toml configuration...${NC}"
        cargo deny init
    fi
    
    # Run all deny checks
    if cargo deny check > "${REPORTS_DIR}/cargo-deny-${TIMESTAMP}.txt" 2>&1; then
        echo -e "${GREEN}✓ All cargo-deny checks passed${NC}"
        return 0
    else
        echo -e "${RED}✗ Cargo-deny checks failed${NC}"
        return 1
    fi
}

# Check for known security patterns in code
run_code_analysis() {
    echo -e "${GREEN}Running code security analysis...${NC}"
    
    cd "${WORKSPACE_ROOT}"
    
    # Check for common security anti-patterns
    local issues_found=0
    
    # Check for unsafe blocks
    echo "Checking for unsafe blocks..."
    if grep -r "unsafe" --include="*.rs" crates/ > "${REPORTS_DIR}/unsafe-blocks-${TIMESTAMP}.txt"; then
        echo -e "${YELLOW}⚠ Found unsafe blocks (review required)${NC}"
        issues_found=1
    fi
    
    # Check for TODO/FIXME related to security
    echo "Checking for security TODOs..."
    if grep -r -i "todo.*secur\|fixme.*secur\|xxx.*secur" --include="*.rs" crates/ > "${REPORTS_DIR}/security-todos-${TIMESTAMP}.txt"; then
        echo -e "${YELLOW}⚠ Found security-related TODOs${NC}"
        issues_found=1
    fi
    
    # Check for hardcoded secrets patterns
    echo "Checking for potential hardcoded secrets..."
    if grep -r -E "(password|secret|key|token).*=.*[\"'][^\"']+[\"']" --include="*.rs" crates/ > "${REPORTS_DIR}/potential-secrets-${TIMESTAMP}.txt"; then
        echo -e "${YELLOW}⚠ Found potential hardcoded secrets${NC}"
        issues_found=1
    fi
    
    if [ $issues_found -eq 0 ]; then
        echo -e "${GREEN}✓ No security issues found in code analysis${NC}"
        return 0
    else
        echo -e "${YELLOW}⚠ Security issues found (see reports)${NC}"
        return 1
    fi
}

# Generate security report summary
generate_summary() {
    echo -e "${GREEN}Generating security report summary...${NC}"
    
    local summary_file="${REPORTS_DIR}/security-summary-${TIMESTAMP}.md"
    
    cat > "${summary_file}" << EOF
# Security Report Summary

**Generated:** $(date)
**Workspace:** ${WORKSPACE_ROOT}

## Vulnerability Scan Results

EOF

    if [ -f "${REPORTS_DIR}/security-audit-${TIMESTAMP}.json" ]; then
        echo "### cargo-audit Results: ✅ PASSED" >> "${summary_file}"
    else
        echo "### cargo-audit Results: ❌ FAILED" >> "${summary_file}"
        echo "See: security-audit-${TIMESTAMP}.txt" >> "${summary_file}"
    fi
    
    echo "" >> "${summary_file}"
    
    if [ -f "${REPORTS_DIR}/cargo-deny-${TIMESTAMP}.txt" ]; then
        if grep -q "error" "${REPORTS_DIR}/cargo-deny-${TIMESTAMP}.txt"; then
            echo "### cargo-deny Results: ❌ FAILED" >> "${summary_file}"
        else
            echo "### cargo-deny Results: ✅ PASSED" >> "${summary_file}"
        fi
    fi
    
    echo "" >> "${summary_file}"
    echo "### Code Analysis Results" >> "${summary_file}"
    
    if [ -f "${REPORTS_DIR}/unsafe-blocks-${TIMESTAMP}.txt" ]; then
        echo "- ⚠️ Unsafe blocks found (review required)" >> "${summary_file}"
    fi
    
    if [ -f "${REPORTS_DIR}/security-todos-${TIMESTAMP}.txt" ]; then
        echo "- ⚠️ Security TODOs found" >> "${summary_file}"
    fi
    
    if [ -f "${REPORTS_DIR}/potential-secrets-${TIMESTAMP}.txt" ]; then
        echo "- ⚠️ Potential hardcoded secrets found" >> "${summary_file}"
    fi
    
    echo "" >> "${summary_file}"
    echo "## Files Generated" >> "${summary_file}"
    echo "" >> "${summary_file}"
    ls -la "${REPORTS_DIR}"/*${TIMESTAMP}* | sed 's/^/- /' >> "${summary_file}"
    
    echo -e "${GREEN}Summary report: ${summary_file}${NC}"
}

# Main execution
main() {
    echo -e "${GREEN}Starting security vulnerability scan...${NC}"
    
    check_dependencies
    setup_reports
    
    local exit_code=0
    
    # Run security checks
    run_security_audit || exit_code=1
    run_cargo_deny || exit_code=1
    run_code_analysis || exit_code=1
    
    # Generate summary
    generate_summary
    
    if [ $exit_code -eq 0 ]; then
        echo -e "${GREEN}✅ Security scan completed successfully${NC}"
    else
        echo -e "${RED}❌ Security scan found issues${NC}"
    fi
    
    exit $exit_code
}

# Run main function
main "$@"