#!/usr/bin/env bash
set -e
export CUDA_HOME=/usr/local/cuda-13.3
export PATH="$CUDA_HOME/bin:$HOME/.local/bin:$PATH"
cd /mnt/c/Users/ugcir/Downloads/PA8
echo "===== HYBRID (custom c1 + cuDNN), batch 256, benchmark on ====="
python3 pytorch_net.py --profile --batch 2000 --custom --cudnn-benchmark --iters 20 2>&1 | tail -13
echo ""
echo "===== PURE cuDNN, batch 256, benchmark on ====="
python3 pytorch_net.py --profile --batch 2000 --cudnn-benchmark --iters 20 2>&1 | tail -13
