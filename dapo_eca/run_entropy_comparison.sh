#!/bin/bash

# Load required modules
module load gcc/13.3.1-p20240614  # PyTorch requires GCC 9+
module load cuda/12.6.1

# Run the entropy comparison script
python entropy_comparison.py "$@"
