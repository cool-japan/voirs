#!/bin/bash

# VoiRS Docker Helper Script
# This script provides convenient commands for managing VoiRS Docker containers

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}=== $1 ===${NC}"
}

# Check if Docker is running
check_docker() {
    if ! docker info > /dev/null 2>&1; then
        print_error "Docker is not running. Please start Docker first."
        exit 1
    fi
}

# Check if docker-compose is available
check_compose() {
    if ! command -v docker-compose > /dev/null 2>&1; then
        print_error "docker-compose is not installed. Please install it first."
        exit 1
    fi
}

# Show help
show_help() {
    cat << EOF
VoiRS Docker Helper Script

Usage: $0 [COMMAND] [OPTIONS]

Commands:
    build           Build Docker images
    start           Start services
    stop            Stop services
    restart         Restart services
    logs            View logs
    shell           Open shell in container
    clean           Clean up Docker resources
    dev             Start development environment
    prod            Start production environment
    health          Check service health
    backup          Backup data
    restore         Restore data
    ssl             Generate SSL certificates
    help            Show this help message

Options:
    -s, --service   Specify service name (default: voirs)
    -f, --follow    Follow logs (for logs command)
    -d, --detach    Run in background
    -v, --verbose   Verbose output

Examples:
    $0 build                    # Build all images
    $0 start -d                 # Start services in background
    $0 logs -f                  # Follow logs
    $0 shell -s voirs           # Open shell in voirs container
    $0 dev                      # Start development environment
    $0 prod                     # Start production environment
    $0 clean                    # Clean up Docker resources

EOF
}

# Build images
build_images() {
    print_header "Building VoiRS Docker Images"
    
    print_status "Building production image..."
    docker-compose build voirs
    
    print_status "Building development image..."
    docker-compose build voirs-dev
    
    print_status "Build completed successfully"
}

# Start services
start_services() {
    local detach=""
    local profile=""
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            -d|--detach)
                detach="-d"
                shift
                ;;
            --profile)
                profile="--profile $2"
                shift 2
                ;;
            *)
                shift
                ;;
        esac
    done
    
    print_header "Starting VoiRS Services"
    docker-compose $profile up $detach
}

# Stop services
stop_services() {
    print_header "Stopping VoiRS Services"
    docker-compose down
    print_status "Services stopped"
}

# Restart services
restart_services() {
    print_header "Restarting VoiRS Services"
    docker-compose restart
    print_status "Services restarted"
}

# View logs
view_logs() {
    local service="voirs"
    local follow=""
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            -s|--service)
                service="$2"
                shift 2
                ;;
            -f|--follow)
                follow="-f"
                shift
                ;;
            *)
                shift
                ;;
        esac
    done
    
    print_header "Viewing logs for $service"
    docker-compose logs $follow $service
}

# Open shell in container
open_shell() {
    local service="voirs"
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            -s|--service)
                service="$2"
                shift 2
                ;;
            *)
                shift
                ;;
        esac
    done
    
    print_header "Opening shell in $service container"
    docker-compose exec $service /bin/bash
}

# Clean up Docker resources
clean_resources() {
    print_header "Cleaning up Docker resources"
    
    print_status "Stopping all containers..."
    docker-compose down
    
    print_status "Removing unused images..."
    docker image prune -f
    
    print_status "Removing unused volumes..."
    docker volume prune -f
    
    print_status "Removing unused networks..."
    docker network prune -f
    
    print_status "Cleanup completed"
}

# Start development environment
start_dev() {
    print_header "Starting Development Environment"
    docker-compose --profile dev up voirs-dev
}

# Start production environment
start_prod() {
    print_header "Starting Production Environment"
    
    # Check if .env file exists
    if [[ ! -f .env ]]; then
        print_warning ".env file not found. Creating template..."
        cat > .env << EOF
# Database configuration
POSTGRES_DB=voirs
POSTGRES_USER=voirs
POSTGRES_PASSWORD=change_this_password

# Redis configuration
REDIS_PASSWORD=change_this_redis_password

# Application configuration
RUST_LOG=info
VOIRS_ENV=production
EOF
        print_warning "Please edit .env file with your configuration before starting production."
        return 1
    fi
    
    docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
    print_status "Production environment started"
}

# Check service health
check_health() {
    print_header "Checking Service Health"
    
    # Check container status
    print_status "Container status:"
    docker-compose ps
    
    # Check health of each service
    print_status "Health checks:"
    for service in voirs redis postgres nginx; do
        if docker-compose ps $service > /dev/null 2>&1; then
            health=$(docker-compose ps $service | awk 'NR==2 {print $4}')
            echo "  $service: $health"
        fi
    done
}

# Backup data
backup_data() {
    print_header "Backing up VoiRS data"
    
    local backup_dir="./backups/$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$backup_dir"
    
    # Backup PostgreSQL
    if docker-compose ps postgres > /dev/null 2>&1; then
        print_status "Backing up PostgreSQL..."
        docker-compose exec postgres pg_dump -U voirs voirs > "$backup_dir/postgres.sql"
    fi
    
    # Backup Redis
    if docker-compose ps redis > /dev/null 2>&1; then
        print_status "Backing up Redis..."
        docker-compose exec redis redis-cli save
        docker cp $(docker-compose ps -q redis):/data/dump.rdb "$backup_dir/redis.rdb"
    fi
    
    # Backup application data
    if [[ -d ./data ]]; then
        print_status "Backing up application data..."
        cp -r ./data "$backup_dir/app_data"
    fi
    
    print_status "Backup completed: $backup_dir"
}

# Restore data
restore_data() {
    local backup_dir="$1"
    
    if [[ -z "$backup_dir" ]]; then
        print_error "Please specify backup directory"
        return 1
    fi
    
    if [[ ! -d "$backup_dir" ]]; then
        print_error "Backup directory does not exist: $backup_dir"
        return 1
    fi
    
    print_header "Restoring VoiRS data from $backup_dir"
    
    # Restore PostgreSQL
    if [[ -f "$backup_dir/postgres.sql" ]]; then
        print_status "Restoring PostgreSQL..."
        docker-compose exec -T postgres psql -U voirs voirs < "$backup_dir/postgres.sql"
    fi
    
    # Restore Redis
    if [[ -f "$backup_dir/redis.rdb" ]]; then
        print_status "Restoring Redis..."
        docker cp "$backup_dir/redis.rdb" $(docker-compose ps -q redis):/data/dump.rdb
        docker-compose restart redis
    fi
    
    # Restore application data
    if [[ -d "$backup_dir/app_data" ]]; then
        print_status "Restoring application data..."
        cp -r "$backup_dir/app_data" ./data
    fi
    
    print_status "Restore completed"
}

# Generate SSL certificates
generate_ssl() {
    print_header "Generating SSL certificates"
    
    mkdir -p docker/nginx/ssl
    
    openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
        -keyout docker/nginx/ssl/server.key \
        -out docker/nginx/ssl/server.crt \
        -subj "/C=US/ST=State/L=City/O=Organization/CN=localhost"
    
    print_status "SSL certificates generated in docker/nginx/ssl/"
}

# Main script logic
main() {
    check_docker
    check_compose
    
    case "${1:-help}" in
        build)
            build_images
            ;;
        start)
            shift
            start_services "$@"
            ;;
        stop)
            stop_services
            ;;
        restart)
            restart_services
            ;;
        logs)
            shift
            view_logs "$@"
            ;;
        shell)
            shift
            open_shell "$@"
            ;;
        clean)
            clean_resources
            ;;
        dev)
            start_dev
            ;;
        prod)
            start_prod
            ;;
        health)
            check_health
            ;;
        backup)
            backup_data
            ;;
        restore)
            shift
            restore_data "$@"
            ;;
        ssl)
            generate_ssl
            ;;
        help|*)
            show_help
            ;;
    esac
}

# Run main function
main "$@"