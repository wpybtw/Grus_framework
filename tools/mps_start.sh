#!/bin/bash
# export CUDA_VISIBLE_DEVICES=0
# sudo nvidia-smi -i 0 -c EXCLUSIVE_PROCESS
# nvidia-cuda-mps-control –d
# export CUDA_VISIBLE_DEVICES=0 # Select GPU 0.
nvidia-smi -i 0 -c EXCLUSIVE_PROCESS # Set GPU 0 to exclusive mode.
nvidia-cuda-mps-control -d # Start the daemon