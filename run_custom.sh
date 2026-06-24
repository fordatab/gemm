#!/usr/bin/env bash
# Helper to run the custom-kernel profiler in WSL with the CUDA toolkit on PATH.
set -e
export CUDA_HOME=/usr/local/cuda-13.3
export PATH="$CUDA_HOME/bin:$HOME/.local/bin:$PATH"
cd /mnt/c/Users/ugcir/Downloads/PA8
echo "ninja: $(command -v ninja)"
echo "nvcc:  $(command -v nvcc)"
python3 pytorch_net.py --profile --batch 2000 --cudnn-benchmark  "$@"

#python3 utils/profile.py --args python3 pytorch_net.py --profile --batch 2000 --custom --iters 20 "$@"
