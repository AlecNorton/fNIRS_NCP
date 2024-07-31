#!/bin/bash
#SBATCH -N 1
#SBATCH -n 10
#SBATCH --mem=5g
#SBATCH -J "GPU-Test - Alec Norton"
#SBATCH -t 12:00:00
#SBATCH --gres=gpu:2
#SBATCH -C A100|V100
nvidia-smi

module load python/3.10.2
python3 -m venv myenv
source myenv/bin/activate
pip install --upgrade pip
pip install tensorflow==2.15.0


module load cuda12.2

python GPU_Test.py