#!/bin/bash
# Usage: ./build_and_run.sh <image_name> <tag>

set -e  # Exit on error

# Validate inputs
if [ -z "$1" ] || [ -z "$2" ]; then
  echo "Usage: $0 <image_name> <tag>"
  exit 1
fi

IMAGE_NAME="$1"
TAG="$2"

PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
CONTAINER_PROJECT_DIR="/home/containeruser"

# Build the Docker image
docker build \
  --build-arg USERID="$(id -u)" \
  --build-arg GROUPID="$(id -g)" \
  --build-arg REPO_DIR="$CONTAINER_PROJECT_DIR" \
  -t "$IMAGE_NAME:$TAG" \
  "$PROJECT_DIR"

# Run the Docker container
docker run -it -h "$IMAGE_NAME" --name "${IMAGE_NAME}_$USER" \
  -u "$(id -u):$(id -g)" \
  --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --runtime=nvidia \
  -v "$PROJECT_DIR:$CONTAINER_PROJECT_DIR" \
  -v /media:/media \
  -w "$CONTAINER_PROJECT_DIR" \
  "$IMAGE_NAME:$TAG"
