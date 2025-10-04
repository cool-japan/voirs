#!/bin/bash

# VoiRS Recognizer CI/CD Setup Script
# This script sets up a complete CI/CD pipeline for the VoiRS Recognizer project

set -euo pipefail

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
GITHUB_DIR="$PROJECT_ROOT/.github"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running in GitHub Actions
is_github_actions() {
    [[ "${GITHUB_ACTIONS:-false}" == "true" ]]
}

# Check if running locally
is_local() {
    ! is_github_actions
}

# Function to check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    local missing_tools=()
    
    # Check for required tools
    if ! command -v git &> /dev/null; then
        missing_tools+=("git")
    fi
    
    if ! command -v cargo &> /dev/null; then
        missing_tools+=("cargo (Rust)")
    fi
    
    if ! command -v docker &> /dev/null; then
        missing_tools+=("docker")
    fi
    
    if ! command -v docker-compose &> /dev/null && ! command -v docker compose &> /dev/null; then
        missing_tools+=("docker-compose")
    fi
    
    if [[ ${#missing_tools[@]} -gt 0 ]]; then
        log_error "Missing required tools: ${missing_tools[*]}"
        log_error "Please install the missing tools and run this script again."
        exit 1
    fi
    
    log_success "All prerequisites are available"
}

# Function to setup GitHub repository secrets (if running locally)
setup_github_secrets() {
    if is_github_actions; then
        log_info "Running in GitHub Actions, skipping secret setup"
        return
    fi
    
    log_info "Setting up GitHub repository secrets..."
    
    # Check if GitHub CLI is available
    if command -v gh &> /dev/null; then
        log_info "GitHub CLI detected. You can set up secrets using 'gh secret set'"
        echo ""
        echo "Required secrets for full CI/CD functionality:"
        echo "  - CARGO_REGISTRY_TOKEN: For publishing to crates.io"
        echo "  - NPM_TOKEN: For publishing WASM packages to npm"
        echo "  - DOCKERHUB_USERNAME: For Docker Hub publishing"
        echo "  - DOCKERHUB_TOKEN: For Docker Hub publishing"
        echo "  - CODECOV_TOKEN: For code coverage reporting (optional)"
        echo ""
        echo "Example commands:"
        echo "  gh secret set CARGO_REGISTRY_TOKEN"
        echo "  gh secret set NPM_TOKEN"
        echo "  gh secret set DOCKERHUB_USERNAME"
        echo "  gh secret set DOCKERHUB_TOKEN"
    else
        log_warning "GitHub CLI not found. Please set up repository secrets manually:"
        echo "1. Go to your repository on GitHub"
        echo "2. Navigate to Settings > Secrets and variables > Actions"
        echo "3. Add the required secrets (see README for details)"
    fi
}

# Function to create configuration files
create_config_files() {
    log_info "Creating configuration files..."
    
    local config_dir="$PROJECT_ROOT/config"
    mkdir -p "$config_dir"
    
    # Create Redis configuration
    cat > "$config_dir/redis.conf" << 'EOF'
# Redis configuration for VoiRS Recognizer
bind 0.0.0.0
port 6379
timeout 0
keepalive 300
maxmemory 256mb
maxmemory-policy allkeys-lru
save 900 1
save 300 10
save 60 10000
EOF
    
    # Create Nginx configuration
    cat > "$config_dir/nginx.conf" << 'EOF'
events {
    worker_connections 1024;
}

http {
    upstream voirs_api {
        server voirs-api:8080;
    }
    
    upstream voirs_wasm {
        server voirs-wasm-dev:8080;
    }
    
    server {
        listen 80;
        server_name localhost;
        
        # API endpoints
        location /api/ {
            proxy_pass http://voirs_api/;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
        
        # WASM demo
        location /demo/ {
            proxy_pass http://voirs_wasm/;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
        }
        
        # Health check
        location /health {
            proxy_pass http://voirs_api/health;
        }
        
        # Default redirect
        location / {
            return 301 /demo/;
        }
    }
}
EOF
    
    # Create Prometheus configuration
    cat > "$config_dir/prometheus.yml" << 'EOF'
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  # - "first_rules.yml"
  # - "second_rules.yml"

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']
  
  - job_name: 'voirs-api'
    static_configs:
      - targets: ['voirs-api:8080']
    metrics_path: '/metrics'
    scrape_interval: 5s
EOF
    
    # Create Loki configuration
    cat > "$config_dir/loki.yml" << 'EOF'
auth_enabled: false

server:
  http_listen_port: 3100

ingester:
  lifecycler:
    address: 127.0.0.1
    ring:
      kvstore:
        store: inmemory
      replication_factor: 1
    final_sleep: 0s
  chunk_idle_period: 1h
  max_chunk_age: 1h
  chunk_target_size: 1048576
  chunk_retain_period: 30s
  max_transfer_retries: 0

schema_config:
  configs:
    - from: 2020-10-24
      store: boltdb-shipper
      object_store: filesystem
      schema: v11
      index:
        prefix: index_
        period: 24h

storage_config:
  boltdb_shipper:
    active_index_directory: /loki/boltdb-shipper-active
    cache_location: /loki/boltdb-shipper-cache
    cache_ttl: 24h
    shared_store: filesystem
  filesystem:
    directory: /loki/chunks

limits_config:
  reject_old_samples: true
  reject_old_samples_max_age: 168h

chunk_store_config:
  max_look_back_period: 0s

table_manager:
  retention_deletes_enabled: false
  retention_period: 0s
EOF
    
    # Create Promtail configuration
    cat > "$config_dir/promtail.yml" << 'EOF'
server:
  http_listen_port: 9080
  grpc_listen_port: 0

positions:
  filename: /tmp/positions.yaml

clients:
  - url: http://loki:3100/loki/api/v1/push

scrape_configs:
  - job_name: containers
    static_configs:
      - targets:
          - localhost
        labels:
          job: containerlogs
          __path__: /var/log/containers/*.log
  
  - job_name: voirs-api
    static_configs:
      - targets:
          - localhost
        labels:
          job: voirs-api
          __path__: /app/logs/*.log
EOF
    
    log_success "Configuration files created in $config_dir"
}

# Function to create database initialization script
create_db_init() {
    log_info "Creating database initialization script..."
    
    local scripts_dir="$PROJECT_ROOT/scripts"
    mkdir -p "$scripts_dir"
    
    cat > "$scripts_dir/init-db.sql" << 'EOF'
-- VoiRS Recognizer Database Initialization
-- This script sets up the database schema for analytics and metadata

-- Create tables for recognition sessions
CREATE TABLE IF NOT EXISTS recognition_sessions (
    id SERIAL PRIMARY KEY,
    session_id UUID UNIQUE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    duration_ms INTEGER,
    audio_length_ms INTEGER,
    model_used VARCHAR(100),
    features_used TEXT[],
    status VARCHAR(50) DEFAULT 'completed',
    error_message TEXT
);

-- Create table for performance metrics
CREATE TABLE IF NOT EXISTS performance_metrics (
    id SERIAL PRIMARY KEY,
    session_id UUID REFERENCES recognition_sessions(session_id),
    metric_name VARCHAR(100) NOT NULL,
    metric_value DECIMAL(10,4),
    unit VARCHAR(20),
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create table for recognition results
CREATE TABLE IF NOT EXISTS recognition_results (
    id SERIAL PRIMARY KEY,
    session_id UUID REFERENCES recognition_sessions(session_id),
    text_result TEXT,
    confidence_score DECIMAL(5,4),
    language_detected VARCHAR(10),
    speaker_count INTEGER,
    segment_start_ms INTEGER,
    segment_end_ms INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_sessions_created_at ON recognition_sessions(created_at);
CREATE INDEX IF NOT EXISTS idx_sessions_model ON recognition_sessions(model_used);
CREATE INDEX IF NOT EXISTS idx_metrics_session ON performance_metrics(session_id);
CREATE INDEX IF NOT EXISTS idx_metrics_name ON performance_metrics(metric_name);
CREATE INDEX IF NOT EXISTS idx_results_session ON recognition_results(session_id);

-- Create a view for session summaries
CREATE OR REPLACE VIEW session_summaries AS
SELECT 
    rs.session_id,
    rs.created_at,
    rs.duration_ms,
    rs.audio_length_ms,
    rs.model_used,
    rs.status,
    COUNT(rr.id) as result_count,
    AVG(rr.confidence_score) as avg_confidence
FROM recognition_sessions rs
LEFT JOIN recognition_results rr ON rs.session_id = rr.session_id
GROUP BY rs.session_id, rs.created_at, rs.duration_ms, rs.audio_length_ms, rs.model_used, rs.status;

-- Insert sample data (for testing)
INSERT INTO recognition_sessions (session_id, duration_ms, audio_length_ms, model_used) 
VALUES 
    (gen_random_uuid(), 1500, 10000, 'whisper-base'),
    (gen_random_uuid(), 2300, 15000, 'whisper-small')
ON CONFLICT DO NOTHING;

COMMIT;
EOF
    
    log_success "Database initialization script created"
}

# Function to create environment file template
create_env_template() {
    log_info "Creating environment file template..."
    
    cat > "$PROJECT_ROOT/.env.example" << 'EOF'
# VoiRS Recognizer Environment Configuration
# Copy this file to .env and customize the values

# Application Configuration
VERSION=0.1.0-alpha.2
RUST_LOG=info

# Service Ports
API_PORT=8080
WASM_PORT=8081
DEV_PORT=3000
DEV_API_PORT=8000
REDIS_PORT=6379
POSTGRES_PORT=5432
HTTP_PORT=80
HTTPS_PORT=443
PROMETHEUS_PORT=9090
GRAFANA_PORT=3001
LOKI_PORT=3100

# API Configuration
MAX_UPLOAD_SIZE=100MB
CORS_ORIGIN=*

# Database Configuration
POSTGRES_DB=voirs
POSTGRES_USER=voirs
POSTGRES_PASSWORD=voirs123

# Monitoring Configuration
GRAFANA_USER=admin
GRAFANA_PASSWORD=admin

# Paths (for volume mounting)
MODELS_PATH=./models

# Build Configuration
BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ')
VCS_REF=$(git rev-parse HEAD)
EOF
    
    # Create default .env if it doesn't exist
    if [[ ! -f "$PROJECT_ROOT/.env" ]]; then
        cp "$PROJECT_ROOT/.env.example" "$PROJECT_ROOT/.env"
        log_success "Default .env file created"
    else
        log_info ".env file already exists, skipping creation"
    fi
}

# Function to setup development environment
setup_development() {
    log_info "Setting up development environment..."
    
    # Install development tools
    if command -v cargo &> /dev/null; then
        log_info "Installing Rust development tools..."
        cargo install cargo-watch cargo-audit cargo-deny || true
        
        # Install wasm-pack for WASM development
        if ! command -v wasm-pack &> /dev/null; then
            log_info "Installing wasm-pack..."
            curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh || true
        fi
    fi
    
    # Setup Git hooks
    local hooks_dir="$PROJECT_ROOT/.git/hooks"
    if [[ -d "$hooks_dir" ]]; then
        log_info "Setting up Git hooks..."
        
        # Pre-commit hook
        cat > "$hooks_dir/pre-commit" << 'EOF'
#!/bin/bash
# Pre-commit hook for VoiRS Recognizer

set -e

echo "Running pre-commit checks..."

# Check formatting
echo "Checking code formatting..."
cargo fmt --all -- --check

# Run clippy
echo "Running clippy..."
cargo clippy --all-targets --all-features -- -D warnings

# Run tests
echo "Running tests..."
cargo test --all-features

echo "Pre-commit checks passed!"
EOF
        chmod +x "$hooks_dir/pre-commit"
        
        log_success "Git hooks installed"
    fi
}

# Function to validate the CI/CD setup
validate_setup() {
    log_info "Validating CI/CD setup..."
    
    local validation_errors=()
    
    # Check if GitHub workflows exist
    if [[ ! -f "$GITHUB_DIR/workflows/ci.yml" ]]; then
        validation_errors+=("Missing CI workflow")
    fi
    
    if [[ ! -f "$GITHUB_DIR/workflows/release.yml" ]]; then
        validation_errors+=("Missing release workflow")
    fi
    
    if [[ ! -f "$GITHUB_DIR/workflows/performance.yml" ]]; then
        validation_errors+=("Missing performance workflow")
    fi
    
    # Check if Docker files exist
    if [[ ! -f "$PROJECT_ROOT/Dockerfile" ]]; then
        validation_errors+=("Missing Dockerfile")
    fi
    
    if [[ ! -f "$PROJECT_ROOT/docker-compose.yml" ]]; then
        validation_errors+=("Missing docker-compose.yml")
    fi
    
    # Check if configuration files exist
    if [[ ! -d "$PROJECT_ROOT/config" ]]; then
        validation_errors+=("Missing config directory")
    fi
    
    # Report validation results
    if [[ ${#validation_errors[@]} -gt 0 ]]; then
        log_error "Validation failed with the following errors:"
        for error in "${validation_errors[@]}"; do
            echo "  - $error"
        done
        return 1
    else
        log_success "CI/CD setup validation passed!"
        return 0
    fi
}

# Function to display usage information
show_usage() {
    echo "VoiRS Recognizer CI/CD Setup Script"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --help, -h          Show this help message"
    echo "  --dev-only          Setup development environment only"
    echo "  --validate-only     Only validate existing setup"
    echo "  --skip-secrets      Skip GitHub secrets setup"
    echo "  --verbose, -v       Enable verbose output"
    echo ""
    echo "Examples:"
    echo "  $0                  Full CI/CD setup"
    echo "  $0 --dev-only       Development setup only"
    echo "  $0 --validate-only  Validate existing setup"
    echo ""
}

# Function to run the complete setup
run_setup() {
    local dev_only=false
    local validate_only=false
    local skip_secrets=false
    local verbose=false
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --help|-h)
                show_usage
                exit 0
                ;;
            --dev-only)
                dev_only=true
                shift
                ;;
            --validate-only)
                validate_only=true
                shift
                ;;
            --skip-secrets)
                skip_secrets=true
                shift
                ;;
            --verbose|-v)
                verbose=true
                set -x
                shift
                ;;
            *)
                log_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done
    
    echo "============================================================"
    echo "VoiRS Recognizer CI/CD Setup"
    echo "============================================================"
    echo ""
    
    # Check prerequisites
    check_prerequisites
    
    if [[ "$validate_only" == "true" ]]; then
        validate_setup
        exit $?
    fi
    
    # Create directory structure
    mkdir -p "$GITHUB_DIR/workflows"
    mkdir -p "$PROJECT_ROOT/scripts"
    mkdir -p "$PROJECT_ROOT/config"
    mkdir -p "$PROJECT_ROOT/models"
    
    # Setup based on mode
    if [[ "$dev_only" == "true" ]]; then
        setup_development
    else
        # Full setup
        create_config_files
        create_db_init
        create_env_template
        setup_development
        
        if [[ "$skip_secrets" != "true" ]]; then
            setup_github_secrets
        fi
    fi
    
    # Validate setup
    if validate_setup; then
        echo ""
        echo "============================================================"
        log_success "CI/CD setup completed successfully!"
        echo "============================================================"
        echo ""
        echo "Next steps:"
        echo "1. Review and customize the .env file"
        echo "2. Set up GitHub repository secrets (if not done already)"
        echo "3. Commit and push the CI/CD configuration"
        echo "4. Test the pipeline with a pull request"
        echo ""
        echo "To start development:"
        echo "  docker-compose --profile development up"
        echo ""
        echo "To start production environment:"
        echo "  docker-compose --profile production up -d"
        echo ""
    else
        log_error "Setup validation failed. Please check the errors above."
        exit 1
    fi
}

# Run the main setup function
run_setup "$@"