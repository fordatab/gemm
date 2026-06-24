#!/usr/bin/env bash
# Print the cuDNN kernel name PyTorch actually runs for conv1 (proof of algorithm).
set -e
export CUDA_HOME=/usr/local/cuda-13.3
export PATH="$CUDA_HOME/bin:$HOME/.local/bin:$PATH"
cd /mnt/c/Users/ugcir/Downloads/PA8
python3 cudnn_probe.py --batch "${1:-2000}"
