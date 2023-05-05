#!/bin/bash

#SBATCH --job-name=eval          # Job name
#SBATCH --output=job.%j.out      # Name of output file (%j expands to jobId)
#SBATCH --cpus-per-task=8        # Schedule 8 cores (includes hyperthreading)
#SBATCH --gres=gpu               # Schedule GPU
#SBATCH --time=06:00:00          # Run time (hh:mm:ss) - run for one hour max
#SBATCH --account=students       # Run time (hh:mm:ss) - run for one hour max
#SBATCH --partition=brown        # Run on either the Red or Brown queue

echo "Running on $(hostname):"
nvidia-smi

# Print out the hostname of the node the job is running on
hostname

# Run Training Scripts
poetry run python src/eval.py -M alexnet -V v0
poetry run python src/eval.py -M googlenet -V v0
poetry run python src/eval.py -M resnet18 -V v0
poetry run python src/eval.py -M resnet50 -V v0
poetry run python src/eval.py -M densenet121 -V v0
poetry run python src/eval.py -M mobilenet_v3_small -V v0
poetry run python src/eval.py -M vit_b_16 -V v0
poetry run python src/eval.py -M efficientnet_v2_s -V v0
poetry run python src/eval.py -M convnext_tiny -V v0

poetry run python src/eval.py -M r2plus1d_18 -V v0
poetry run python src/eval.py -M x3d_s -V v0
poetry run python src/eval.py -M slow_r50 -V v0
poetry run python src/eval.py -M slowfast_r50 -V v0
