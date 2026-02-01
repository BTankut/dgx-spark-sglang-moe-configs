#!/bin/bash
# Build and deploy SGLang container for DGX Spark cluster
#
# Usage:
#   ./build-and-deploy.sh                    # Build only
#   ./build-and-deploy.sh --deploy           # Build and deploy to all nodes
#   ./build-and-deploy.sh --deploy-only      # Deploy existing image (no build)
#   ./build-and-deploy.sh --nodes "node1 node2"  # Deploy to specific nodes

set -e

# Configuration
IMAGE_NAME="sglang-spark-glm47"
IMAGE_TAG="latest"
CONTAINER_NAME="sglang_node"
DEFAULT_NODES="dgxnode1 dgxnode2 dgxnode3 dgxnode4"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[OK]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Parse arguments
BUILD=true
DEPLOY=false
NODES="$DEFAULT_NODES"

while [[ $# -gt 0 ]]; do
    case $1 in
        --deploy)
            DEPLOY=true
            shift
            ;;
        --deploy-only)
            BUILD=false
            DEPLOY=true
            shift
            ;;
        --nodes)
            NODES="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --deploy        Build and deploy to all nodes"
            echo "  --deploy-only   Deploy existing image (skip build)"
            echo "  --nodes \"n1 n2\" Specify target nodes"
            echo "  --help          Show this help"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Build
if [ "$BUILD" = true ]; then
    log_info "Building Docker image: ${IMAGE_NAME}:${IMAGE_TAG}"

    # Check if we're in the right directory
    if [ ! -f "Dockerfile" ]; then
        log_error "Dockerfile not found. Run from repo root."
        exit 1
    fi

    docker build -t ${IMAGE_NAME}:${IMAGE_TAG} .
    log_success "Image built successfully"

    # Show image info
    docker images ${IMAGE_NAME}:${IMAGE_TAG}
fi

# Deploy
if [ "$DEPLOY" = true ]; then
    log_info "Deploying to nodes: $NODES"

    # Save image to temp file
    TEMP_TAR="/tmp/${IMAGE_NAME}-${IMAGE_TAG}.tar"
    log_info "Saving image to $TEMP_TAR"
    docker save ${IMAGE_NAME}:${IMAGE_TAG} > "$TEMP_TAR"

    TAR_SIZE=$(du -h "$TEMP_TAR" | cut -f1)
    log_info "Image size: $TAR_SIZE"

    # Deploy to each node
    for node in $NODES; do
        log_info "Deploying to $node..."

        # Check connectivity
        if ! ssh -o ConnectTimeout=5 "$node" "echo ok" &>/dev/null; then
            log_error "$node is not reachable, skipping"
            continue
        fi

        # Stop existing container if running
        ssh "$node" "docker stop $CONTAINER_NAME 2>/dev/null || true"
        ssh "$node" "docker rm $CONTAINER_NAME 2>/dev/null || true"

        # Transfer and load image
        log_info "  Transferring image..."
        cat "$TEMP_TAR" | ssh "$node" "docker load"

        # Start new container
        log_info "  Starting container..."
        ssh "$node" "docker run -d --name $CONTAINER_NAME \
            --network host --ipc=host --gpus all \
            --ulimit memlock=-1 --ulimit stack=67108864 \
            --device=/dev/infiniband/uverbs0 \
            --device=/dev/infiniband/uverbs1 \
            --device=/dev/infiniband/uverbs2 \
            --device=/dev/infiniband/uverbs3 \
            -v ~/.cache/huggingface:/root/.cache/huggingface \
            ${IMAGE_NAME}:${IMAGE_TAG}"

        log_success "$node deployed"
    done

    # Cleanup
    rm -f "$TEMP_TAR"
    log_success "Deployment complete"

    # Verify
    echo ""
    log_info "Verifying deployment:"
    for node in $NODES; do
        status=$(ssh "$node" "docker ps --filter name=$CONTAINER_NAME --format '{{.Status}}'" 2>/dev/null)
        if [[ $status == Up* ]]; then
            echo -e "  $node: ${GREEN}$status${NC}"
        else
            echo -e "  $node: ${RED}Not running${NC}"
        fi
    done
fi

echo ""
log_info "Next steps:"
echo "  1. Start the cluster: sglang-start"
echo "  2. Or manually: docker exec sglang_node python3 -m sglang.launch_server ..."
