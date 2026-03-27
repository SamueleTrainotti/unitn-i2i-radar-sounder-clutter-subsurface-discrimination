#!/bin/bash

# Wait for NVIDIA drivers to initialize
echo "Checking for NVIDIA GPU availability..."
max_retries=30
retry_delay=5

for ((i=1; i<=$max_retries; i++)); do
    if nvidia-smi >/dev/null 2>&1; then
        echo "NVIDIA drivers initialized successfully!"
        exit 0
    fi
    echo "Attempt $i/$max_retries: NVIDIA drivers not ready yet. Retrying in $retry_delay seconds..."
    sleep $retry_delay
    
    if [ $i -eq $max_retries ]; then
        echo "ERROR: NVIDIA drivers failed to initialize after $max_retries attempts."
        exit 1
    fi
done