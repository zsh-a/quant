#!/bin/bash
# Deployment script for Quant Trading Platform

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICES=(redis api celery_worker frontend)
VALID_LOG_SERVICES=(redis api celery_worker frontend)

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

compose_cmd() {
    if command -v docker-compose >/dev/null 2>&1; then
        docker-compose "$@"
    else
        docker compose "$@"
    fi
}

ensure_requirements() {
    if ! command -v docker >/dev/null 2>&1; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi

    if ! command -v docker-compose >/dev/null 2>&1 && ! docker compose version >/dev/null 2>&1; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
}

ensure_directories() {
    mkdir -p "$ROOT_DIR/data" "$ROOT_DIR/config" "$ROOT_DIR/logs"
}

print_access_points() {
    cat <<'EOF'
Access points:
  - Frontend:  http://localhost
  - API:       http://localhost:8000
  - API Docs:  http://localhost:8000/docs

Core services:
  - redis
  - api
  - celery_worker
  - frontend
EOF
}

validate_log_service() {
    local service="$1"
    local candidate
    for candidate in "${VALID_LOG_SERVICES[@]}"; do
        if [ "$candidate" = "$service" ]; then
            return 0
        fi
    done

    print_error "Unknown service: $service"
    echo "Valid services: ${VALID_LOG_SERVICES[*]}"
    exit 1
}

cmd="${1:-up}"

ensure_requirements
ensure_directories

case "$cmd" in
    build)
        print_info "Building core service images..."
        compose_cmd build "${SERVICES[@]}"
        ;;

    up)
        print_info "Building core service images..."
        compose_cmd build "${SERVICES[@]}"
        print_info "Starting core services..."
        compose_cmd up -d "${SERVICES[@]}"
        print_info "Deployment started."
        compose_cmd ps "${SERVICES[@]}"
        print_access_points
        ;;

    down)
        print_info "Stopping core services..."
        compose_cmd down
        ;;

    restart)
        print_info "Restarting core services..."
        compose_cmd restart "${SERVICES[@]}"
        compose_cmd ps "${SERVICES[@]}"
        ;;

    logs)
        service="${2:-}"
        if [ -n "$service" ]; then
            validate_log_service "$service"
            compose_cmd logs -f "$service"
        else
            compose_cmd logs -f "${SERVICES[@]}"
        fi
        ;;

    status)
        compose_cmd ps "${SERVICES[@]}"
        echo
        print_access_points
        ;;

    *)
        cat <<'EOF'
Usage: ./deploy.sh {build|up|down|restart|logs [service]|status}

Commands:
  build    Build the core service images
  up       Build and start the core services in the background
  down     Stop and remove the compose stack
  restart  Restart the core services
  logs     Follow logs for all core services or a specific service
  status   Show compose status and access points
EOF
        exit 1
        ;;
esac
