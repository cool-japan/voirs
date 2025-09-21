#!/bin/bash

# VoiRS Feedback System Deployment Script
# This script provides automated deployment for different environments

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DEFAULT_ENV="development"
DEFAULT_REGISTRY="docker.io/voirs"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Show usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS] COMMAND

VoiRS Feedback System Deployment Script

COMMANDS:
    build           Build Docker image
    push            Push Docker image to registry
    deploy-local    Deploy using Docker Compose
    deploy-k8s      Deploy to Kubernetes
    cleanup         Clean up resources
    logs            Show deployment logs
    status          Show deployment status

OPTIONS:
    -e, --env ENV           Environment (development, staging, production) [default: $DEFAULT_ENV]
    -r, --registry REGISTRY Docker registry [default: $DEFAULT_REGISTRY]
    -t, --tag TAG          Image tag [default: latest]
    -n, --namespace NS      Kubernetes namespace [default: voirs]
    -h, --help             Show this help message

EXAMPLES:
    $0 build -e production -t v1.0.0
    $0 deploy-local
    $0 deploy-k8s -e staging -n voirs-staging
    $0 cleanup -e development

EOF
}

# Parse command line arguments
ENV="$DEFAULT_ENV"
REGISTRY="$DEFAULT_REGISTRY"
TAG="latest"
NAMESPACE="voirs"
COMMAND=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--env)
            ENV="$2"
            shift 2
            ;;
        -r|--registry)
            REGISTRY="$2"
            shift 2
            ;;
        -t|--tag)
            TAG="$2"
            shift 2
            ;;
        -n|--namespace)
            NAMESPACE="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        build|push|deploy-local|deploy-k8s|cleanup|logs|status)
            COMMAND="$1"
            shift
            ;;
        *)
            log_error "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Validate environment
if [[ ! "$ENV" =~ ^(development|staging|production)$ ]]; then
    log_error "Invalid environment: $ENV. Must be development, staging, or production"
    exit 1
fi

# Set image name
IMAGE_NAME="${REGISTRY}/feedback:${TAG}"

# Build Docker image
build_image() {
    log_info "Building Docker image: $IMAGE_NAME"
    
    cd "$PROJECT_ROOT"
    
    # Build with BuildKit for better caching
    DOCKER_BUILDKIT=1 docker build \
        --platform linux/amd64,linux/arm64 \
        --tag "$IMAGE_NAME" \
        --file Dockerfile \
        --build-arg BUILD_ENV="$ENV" \
        --build-arg VERSION="$TAG" \
        ../..
    
    log_info "Successfully built image: $IMAGE_NAME"
}

# Push Docker image
push_image() {
    log_info "Pushing Docker image: $IMAGE_NAME"
    
    docker push "$IMAGE_NAME"
    
    log_info "Successfully pushed image: $IMAGE_NAME"
}

# Deploy using Docker Compose
deploy_local() {
    log_info "Deploying locally using Docker Compose (env: $ENV)"
    
    cd "$PROJECT_ROOT"
    
    # Set environment variables
    export VOIRS_ENV="$ENV"
    export VOIRS_IMAGE="$IMAGE_NAME"
    export VOIRS_TAG="$TAG"
    
    # Create network if it doesn't exist
    docker network create voirs-network 2>/dev/null || true
    
    # Deploy with appropriate compose file
    if [[ "$ENV" == "development" ]]; then
        docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d
    else
        docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
    fi
    
    log_info "Waiting for services to be ready..."
    sleep 30
    
    # Health check
    if curl -f http://localhost:8080/health >/dev/null 2>&1; then
        log_info "Deployment successful! Service is running on http://localhost:8080"
    else
        log_warn "Service may not be fully ready yet. Check logs with: $0 logs"
    fi
}

# Deploy to Kubernetes
deploy_k8s() {
    log_info "Deploying to Kubernetes (env: $ENV, namespace: $NAMESPACE)"
    
    # Check if kubectl is available
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl is not installed or not in PATH"
        exit 1
    fi
    
    cd "$PROJECT_ROOT"
    
    # Create namespace if it doesn't exist
    kubectl create namespace "$NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -
    
    # Apply Kubernetes manifests
    kubectl apply -f k8s/ -n "$NAMESPACE"
    
    # Update image in deployment
    kubectl set image deployment/voirs-feedback voirs-feedback="$IMAGE_NAME" -n "$NAMESPACE"
    
    # Wait for rollout
    kubectl rollout status deployment/voirs-feedback -n "$NAMESPACE" --timeout=300s
    
    # Get service information
    SERVICE_IP=$(kubectl get service voirs-feedback-loadbalancer -n "$NAMESPACE" -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "pending")
    
    log_info "Deployment successful!"
    log_info "Service URL: http://$SERVICE_IP (if LoadBalancer IP is available)"
    log_info "To check status: kubectl get pods -n $NAMESPACE"
}

# Cleanup resources
cleanup() {
    log_info "Cleaning up resources (env: $ENV)"
    
    case "$ENV" in
        development|staging)
            # Local cleanup
            cd "$PROJECT_ROOT"
            docker-compose down -v --remove-orphans
            docker system prune -f --volumes
            ;;
        production)
            log_warn "Production cleanup requires manual confirmation"
            read -p "Are you sure you want to cleanup production resources? (yes/no): " -r
            if [[ $REPLY == "yes" ]]; then
                kubectl delete -f k8s/ -n "$NAMESPACE" || true
                kubectl delete namespace "$NAMESPACE" || true
            else
                log_info "Cleanup cancelled"
            fi
            ;;
    esac
    
    log_info "Cleanup completed"
}

# Show logs
show_logs() {
    log_info "Showing deployment logs (env: $ENV)"
    
    if [[ "$ENV" == "development" ]]; then
        cd "$PROJECT_ROOT"
        docker-compose logs -f --tail=100 voirs-feedback
    else
        kubectl logs -f deployment/voirs-feedback -n "$NAMESPACE" --tail=100
    fi
}

# Show status
show_status() {
    log_info "Checking deployment status (env: $ENV)"
    
    if [[ "$ENV" == "development" ]]; then
        cd "$PROJECT_ROOT"
        docker-compose ps
        echo
        echo "Health check:"
        curl -s http://localhost:8080/health | jq . 2>/dev/null || curl -s http://localhost:8080/health
    else
        kubectl get pods,services,ingress -n "$NAMESPACE"
        echo
        echo "Health check:"
        kubectl run curl-test --image=curlimages/curl:latest --rm -i --restart=Never -- \
            curl -s http://voirs-feedback-service/health
    fi
}

# Main execution
main() {
    if [[ -z "$COMMAND" ]]; then
        log_error "No command specified"
        usage
        exit 1
    fi
    
    log_info "Starting deployment process..."
    log_info "Environment: $ENV"
    log_info "Registry: $REGISTRY"
    log_info "Tag: $TAG"
    log_info "Image: $IMAGE_NAME"
    
    case "$COMMAND" in
        build)
            build_image
            ;;
        push)
            push_image
            ;;
        deploy-local)
            deploy_local
            ;;
        deploy-k8s)
            deploy_k8s
            ;;
        cleanup)
            cleanup
            ;;
        logs)
            show_logs
            ;;
        status)
            show_status
            ;;
        *)
            log_error "Unknown command: $COMMAND"
            usage
            exit 1
            ;;
    esac
}

# Run main function
main