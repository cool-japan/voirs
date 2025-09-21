# VoiRS Examples Enhanced Build System Makefile
# ==============================================
#
# This Makefile provides convenient targets for building, testing, and validating
# VoiRS examples using the enhanced build system.
#
# Usage:
#   make help           - Show available targets
#   make build          - Build all examples
#   make test           - Build and test all examples
#   make clean          - Clean build artifacts
#   make ci             - Run CI/CD pipeline
#   make quick          - Quick build and test
#   make full           - Full build with comprehensive testing
#
# Configuration:
#   Set PARALLEL_JOBS to control parallelism (default: auto-detect)
#   Set EXAMPLES_PATTERN to filter examples (default: "*")
#   Set CATEGORY to filter by category (default: all)
#   Set PROFILE to use specific build profile (default: development)

.PHONY: help build test clean install check lint format \
        quick full ci docs report benchmark \
        examples-% category-% profile-%

# Default target
.DEFAULT_GOAL := help

# Configuration variables
PARALLEL_JOBS ?= $(shell nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
EXAMPLES_PATTERN ?= "*"
CATEGORY ?= 
PROFILE ?= development
TIMEOUT ?= 60
VERBOSE ?= false

# Directories
EXAMPLES_DIR := $(CURDIR)
REPORTS_DIR := $(EXAMPLES_DIR)/build_reports
LOGS_DIR := $(EXAMPLES_DIR)/build_logs
TOOLS_DIR := $(EXAMPLES_DIR)/../tools

# Tools
BUILD_SCRIPT := $(TOOLS_DIR)/build_examples.sh
BUILD_SYSTEM := $(TOOLS_DIR)/enhanced_build_system.py
BUILD_CONFIG := $(TOOLS_DIR)/build_config.toml

# Colors for output
BOLD := \033[1m
RED := \033[31m
GREEN := \033[32m
YELLOW := \033[33m
BLUE := \033[34m
PURPLE := \033[35m
CYAN := \033[36m
WHITE := \033[37m
RESET := \033[0m

# Timestamp for reports
TIMESTAMP := $(shell date +%Y%m%d_%H%M%S)

##@ Help

help: ## Display this help message
	@echo "$(CYAN)$(BOLD)VoiRS Examples Build System$(RESET)"
	@echo "$(CYAN)==============================$(RESET)"
	@echo ""
	@echo "$(BOLD)Usage:$(RESET)"
	@echo "  make $(GREEN)<target>$(RESET)"
	@echo ""
	@awk 'BEGIN {FS = ":.*##"; printf "$(BOLD)Targets:$(RESET)\n"} /^[a-zA-Z_0-9-]+:.*?##/ { printf "  $(GREEN)%-20s$(RESET) %s\n", $$1, $$2 } /^##@/ { printf "\n$(BOLD)%s:$(RESET)\n", substr($$0, 5) }' $(MAKEFILE_LIST)
	@echo ""
	@echo "$(BOLD)Configuration:$(RESET)"
	@echo "  $(YELLOW)PARALLEL_JOBS$(RESET)    = $(PARALLEL_JOBS)"
	@echo "  $(YELLOW)EXAMPLES_PATTERN$(RESET) = $(EXAMPLES_PATTERN)"
	@echo "  $(YELLOW)CATEGORY$(RESET)         = $(CATEGORY)"
	@echo "  $(YELLOW)PROFILE$(RESET)          = $(PROFILE)"
	@echo "  $(YELLOW)TIMEOUT$(RESET)          = $(TIMEOUT)"
	@echo ""
	@echo "$(BOLD)Examples:$(RESET)"
	@echo "  make build PARALLEL_JOBS=8"
	@echo "  make test CATEGORY=performance"
	@echo "  make examples-*benchmark*"
	@echo "  make category-voice"
	@echo "  make profile-production"

##@ Building

build: setup ## Build all examples
	@echo "$(GREEN)$(BOLD)Building VoiRS Examples$(RESET)"
	@$(BUILD_SCRIPT) --build-only \
		--parallel $(PARALLEL_JOBS) \
		--examples "$(EXAMPLES_PATTERN)" \
		$(if $(CATEGORY),--category $(CATEGORY)) \
		$(if $(filter true,$(VERBOSE)),--verbose)

build-release: ## Build all examples in release mode with optimizations
	@echo "$(GREEN)$(BOLD)Building VoiRS Examples (Release Mode)$(RESET)"
	@$(BUILD_SCRIPT) --build-only --full \
		--parallel $(PARALLEL_JOBS) \
		--examples "$(EXAMPLES_PATTERN)" \
		$(if $(CATEGORY),--category $(CATEGORY))

##@ Testing

test: setup ## Build and test all examples
	@echo "$(BLUE)$(BOLD)Testing VoiRS Examples$(RESET)"
	@$(BUILD_SCRIPT) \
		--parallel $(PARALLEL_JOBS) \
		--timeout $(TIMEOUT) \
		--examples "$(EXAMPLES_PATTERN)" \
		$(if $(CATEGORY),--category $(CATEGORY)) \
		$(if $(filter true,$(VERBOSE)),--verbose) \
		--report "$(REPORTS_DIR)/test_report_$(TIMESTAMP).json"

test-only: ## Run tests without building (assumes examples are already built)
	@echo "$(BLUE)$(BOLD)Running Tests Only$(RESET)"
	@$(BUILD_SCRIPT) --test-only \
		--parallel $(PARALLEL_JOBS) \
		--timeout $(TIMEOUT) \
		--examples "$(EXAMPLES_PATTERN)" \
		$(if $(CATEGORY),--category $(CATEGORY))

##@ Quality Assurance

check: lint test ## Run all quality checks (lint + test)

lint: ## Run linting and formatting checks
	@echo "$(YELLOW)$(BOLD)Running Linting Checks$(RESET)"
	@cargo clippy --all-targets --all-features -- -D warnings
	@cargo fmt -- --check

format: ## Format code according to style guidelines
	@echo "$(YELLOW)$(BOLD)Formatting Code$(RESET)"
	@cargo fmt

audit: ## Run security audit
	@echo "$(YELLOW)$(BOLD)Running Security Audit$(RESET)"
	@cargo audit

##@ Advanced Targets

quick: setup ## Quick build and test (optimized for speed)
	@echo "$(CYAN)$(BOLD)Quick Build and Test$(RESET)"
	@$(BUILD_SCRIPT) --quick \
		--parallel $(PARALLEL_JOBS) \
		--timeout 30 \
		--examples "$(EXAMPLES_PATTERN)" \
		$(if $(CATEGORY),--category $(CATEGORY))

full: setup ## Full build with comprehensive testing
	@echo "$(PURPLE)$(BOLD)Full Build and Test$(RESET)"
	@$(BUILD_SCRIPT) --full \
		--parallel $(PARALLEL_JOBS) \
		--timeout 180 \
		--examples "$(EXAMPLES_PATTERN)" \
		$(if $(CATEGORY),--category $(CATEGORY)) \
		--report "$(REPORTS_DIR)/full_report_$(TIMESTAMP).json"

ci: setup ## Run CI/CD pipeline
	@echo "$(WHITE)$(BOLD)Running CI/CD Pipeline$(RESET)"
	@$(BUILD_SCRIPT) --ci --clean \
		--parallel $(PARALLEL_JOBS) \
		--examples "$(EXAMPLES_PATTERN)" \
		--report "$(REPORTS_DIR)/ci_report_$(TIMESTAMP).json"

benchmark: setup ## Run performance benchmarks
	@echo "$(RED)$(BOLD)Running Performance Benchmarks$(RESET)"
	@$(MAKE) test CATEGORY=performance TIMEOUT=300

##@ Utilities

clean: ## Clean all build artifacts and generated files
	@echo "$(RED)$(BOLD)Cleaning Build Artifacts$(RESET)"
	@cargo clean
	@rm -rf $(REPORTS_DIR)/*.json $(REPORTS_DIR)/*.html
	@rm -rf $(LOGS_DIR)/*.log
	@find $(EXAMPLES_DIR) -name "*.wav" -type f -delete 2>/dev/null || true
	@find $(EXAMPLES_DIR) -name "*_report.json" -type f -delete 2>/dev/null || true
	@find $(EXAMPLES_DIR) -name "*_metrics.txt" -type f -delete 2>/dev/null || true
	@echo "$(GREEN)Artifacts cleaned$(RESET)"

install: ## Install build system tools and dependencies
	@echo "$(BLUE)$(BOLD)Installing Build System$(RESET)"
	@mkdir -p $(TOOLS_DIR) $(REPORTS_DIR) $(LOGS_DIR)
	@chmod +x $(BUILD_SCRIPT) 2>/dev/null || true

setup: ## Set up build environment
	@mkdir -p $(REPORTS_DIR) $(LOGS_DIR)

docs: ## Generate documentation
	@echo "$(CYAN)$(BOLD)Generating Documentation$(RESET)"
	@cargo doc --all --no-deps --open

report: ## Generate detailed build report
	@echo "$(YELLOW)$(BOLD)Generating Build Report$(RESET)"
	@python3 $(BUILD_SYSTEM) --report "$(REPORTS_DIR)/detailed_report_$(TIMESTAMP).json"
	@echo "$(GREEN)Report saved to: $(REPORTS_DIR)/detailed_report_$(TIMESTAMP).json$(RESET)"

##@ Example Filtering

examples-%: ## Build/test examples matching pattern (e.g., examples-*benchmark*)
	@$(MAKE) test EXAMPLES_PATTERN="$*"

category-%: ## Build/test examples in specific category (e.g., category-performance)
	@$(MAKE) test CATEGORY="$*"

profile-%: ## Run with specific profile (e.g., profile-production)
	@$(MAKE) test PROFILE="$*"

##@ Platform Specific

linux: ## Build for Linux platform
	@echo "$(BLUE)$(BOLD)Building for Linux$(RESET)"
	@$(MAKE) build CARGO_TARGET=x86_64-unknown-linux-gnu

windows: ## Build for Windows platform
	@echo "$(BLUE)$(BOLD)Building for Windows$(RESET)"
	@$(MAKE) build CARGO_TARGET=x86_64-pc-windows-msvc

macos: ## Build for macOS platform
	@echo "$(BLUE)$(BOLD)Building for macOS$(RESET)"
	@$(MAKE) build CARGO_TARGET=x86_64-apple-darwin

##@ Development

dev: ## Development build and test (fast iteration)
	@echo "$(GREEN)$(BOLD)Development Build$(RESET)"
	@$(MAKE) quick PROFILE=development VERBOSE=true

watch: ## Watch for changes and rebuild (requires cargo-watch)
	@echo "$(CYAN)$(BOLD)Watching for Changes$(RESET)"
	@cargo watch -x "build --examples" -x "test"

##@ Information

status: ## Show build system status
	@echo "$(BOLD)VoiRS Examples Build Status$(RESET)"
	@echo "=============================="
	@echo ""
	@echo "$(BOLD)Environment:$(RESET)"
	@echo "  Platform: $(shell uname -s) $(shell uname -m)"
	@echo "  Rust:     $(shell rustc --version 2>/dev/null || echo 'Not installed')"
	@echo "  Cargo:    $(shell cargo --version 2>/dev/null || echo 'Not installed')"
	@echo "  CPU:      $(shell nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 'Unknown') cores"
	@echo ""
	@echo "$(BOLD)Configuration:$(RESET)"
	@echo "  Examples Dir:    $(EXAMPLES_DIR)"
	@echo "  Reports Dir:     $(REPORTS_DIR)"
	@echo "  Parallel Jobs:   $(PARALLEL_JOBS)"
	@echo "  Current Profile: $(PROFILE)"
	@echo ""
	@echo "$(BOLD)Recent Reports:$(RESET)"
	@ls -la $(REPORTS_DIR)/*.json 2>/dev/null | tail -5 || echo "  No reports found"

examples-list: ## List all available examples
	@echo "$(BOLD)Available Examples:$(RESET)"
	@grep -A1 '\[\[example\]\]' Cargo.toml | grep 'name =' | sed 's/name = "\(.*\)"/  • \1/' | sort

categories-list: ## List example categories
	@echo "$(BOLD)Example Categories:$(RESET)"
	@echo "  • $(GREEN)performance$(RESET)  - Performance and benchmarking examples"
	@echo "  • $(BLUE)testing$(RESET)      - Testing and validation examples"  
	@echo "  • $(YELLOW)realtime$(RESET)     - Real-time and streaming examples"
	@echo "  • $(PURPLE)spatial$(RESET)      - Spatial audio examples"
	@echo "  • $(CYAN)voice$(RESET)        - Voice cloning and customization examples"
	@echo "  • $(WHITE)production$(RESET)   - Production deployment examples"

##@ Debug

debug-build: ## Build with debug information
	@echo "$(RED)$(BOLD)Debug Build$(RESET)"
	@$(BUILD_SCRIPT) --build-only \
		--parallel 1 \
		--verbose \
		--examples "$(EXAMPLES_PATTERN)"

debug-test: ## Run single test with full debug output
	@echo "$(RED)$(BOLD)Debug Test$(RESET)"
	@$(BUILD_SCRIPT) --test-only \
		--parallel 1 \
		--timeout 300 \
		--verbose \
		--examples "$(EXAMPLES_PATTERN)"

# Pattern rules for dynamic targets
.PHONY: build-% test-% clean-%

build-%:
	@$(MAKE) build EXAMPLES_PATTERN="$*"

test-%:
	@$(MAKE) test EXAMPLES_PATTERN="$*"

clean-%:
	@echo "$(RED)Cleaning artifacts for: $*$(RESET)"
	@find $(EXAMPLES_DIR) -name "$**" -delete 2>/dev/null || true

# Include local overrides if they exist
-include Makefile.local