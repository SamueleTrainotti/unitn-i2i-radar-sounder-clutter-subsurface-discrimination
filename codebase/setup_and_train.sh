#!/bin/bash
# Usage: ./setup_and_train.sh <image_name> <tag>
# Example: ./setup_and_train.sh samuele pix2pix
# This script is meant to be run inside the queue system:
#   rs_qsub.sh samuele_train 2:00:00 bash setup_and_train.sh samuele pix2pix

set -e  # Exit on any error

if [ -z "$1" ] || [ -z "$2" ]; then
  echo "Usage: $0 <image_name> <tag>"
  exit 1
fi

IMAGE_NAME="$1"
TAG="$2"
CONTAINER_NAME="${IMAGE_NAME}_$USER"
PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
CONTAINER_PROJECT_DIR="/home/containeruser"

echo "=============================="
echo "Starting queued GPU training job"
echo "Image: $IMAGE_NAME:$TAG"
echo "Container: $CONTAINER_NAME"
echo "=============================="

# Build the Docker image (if not already built)
if ! docker image inspect "$IMAGE_NAME:$TAG" > /dev/null 2>&1; then
  echo "[INFO] Image not found, building..."
  docker build \
    --build-arg USERID="$(id -u)" \
    --build-arg GROUPID="$(id -g)" \
    --build-arg REPO_DIR="$CONTAINER_PROJECT_DIR" \
    -t "$IMAGE_NAME:$TAG" \
    "$PROJECT_DIR"
else
  echo "[INFO] Docker image already exists, skipping build."
fi

# Check if a container with the same name exists
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}\$"; then
  echo "[INFO] Removing existing container: $CONTAINER_NAME"
  docker rm -f "$CONTAINER_NAME" || true
fi

# Create the output directory on the host to ensure correct permissions
OUTPUT_BASE_DIR="/media/datapart/samueletrainotti/data/runs"
echo "[INFO] Ensuring output directory exists: $OUTPUT_BASE_DIR"
mkdir -p "$OUTPUT_BASE_DIR"

# Start container and run training
echo "[INFO] Launching container..."
docker run --rm -t \
  --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -u "$(id -u):$(id -g)" \
  -v "$PROJECT_DIR:$CONTAINER_PROJECT_DIR" \
  -v /media:/media \
  -w "$CONTAINER_PROJECT_DIR" \
  --name "$CONTAINER_NAME" \
  "$IMAGE_NAME:$TAG" \
  python scripts/train.py -c scripts/config_files/config.yaml

echo "[INFO] Training completed successfully."
