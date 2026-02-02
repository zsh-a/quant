#!/bin/bash
# Deployment script for Quant Trading Platform

set -e

echo "=================================================="
echo "Quant Trading Platform - Deployment Script"
echo "=================================================="

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    print_error "Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Parse command line arguments
MODE=${1:-"dev"}  # dev, prod, or stop

case $MODE in
    dev)
        print_info "Starting in DEVELOPMENT mode..."
        
        # Build images
        print_info "Building Docker images..."
        docker-compose build
        docker image prune -f
        
        # Start services
        print_info "Starting services..."
        docker-compose up -d
        
        # Wait for services to be healthy
        print_info "Waiting for services to be ready..."
        sleep 10
        
        # Show status
        docker-compose ps
        
        print_info "Development environment is ready!"
        echo ""
        echo "Access points:"
        echo "  - Frontend:  http://localhost"
        echo "  - API:       http://localhost:8000"
        echo "  - API Docs:  http://localhost:8000/docs"
        echo "  - Flower:    http://localhost:5555"
        echo "  - Prometheus: http://localhost:9090"
        echo "  - Grafana:   http://localhost:3000 (admin/admin)"
        echo ""
        echo "To view logs: docker-compose logs -f [service_name]"
        echo "To stop:      ./deploy.sh stop"
        ;;
    
    prod)
        print_info "Starting in PRODUCTION mode..."
        
        # Build images with no cache
        print_info "Building Docker images (no cache)..."
        docker-compose build --no-cache
        docker image prune -f
        
        # Start services in detached mode
        print_info "Starting services..."
        docker-compose up -d
        
        # Wait for health checks
        print_info "Waiting for health checks..."
        sleep 15
        
        # Verify services are running
        if ! docker-compose ps | grep -q "Up"; then
            print_error "Some services failed to start. Check logs with: docker-compose logs"
            exit 1
        fi
        
        print_info "Production environment is ready!"
        docker-compose ps
        ;;
    
    stop)
        print_info "Stopping all services..."
        docker-compose down
        print_info "All services stopped."
        ;;
    
    restart)
        print_info "Restarting all services..."
        docker-compose restart
        print_info "All services restarted."
        ;;
    
    logs)
        SERVICE=${2:-""}
        if [ -z "$SERVICE" ]; then
            docker-compose logs -f
        else
            docker-compose logs -f $SERVICE
        fi
        ;;
    
    clean)
        print_warn "This will remove all containers, volumes, and images. Are you sure? (y/N)"
        read -r response
        if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
            print_info "Cleaning up..."
            docker-compose down -v --rmi all
            print_info "Cleanup complete."
        else
            print_info "Cleanup cancelled."
        fi
        ;;
    
    *)
        echo "Usage: $0 {dev|prod|stop|restart|logs|clean}"
        echo ""
        echo "Commands:"
        echo "  dev      - Start in development mode"
        echo "  prod     - Start in production mode"
        echo "  stop     - Stop all services"
        echo "  restart  - Restart all services"
        echo "  logs     - View logs (optionally specify service name)"
        echo "  clean    - Remove all containers, volumes, and images"
        exit 1
        ;;
esac
