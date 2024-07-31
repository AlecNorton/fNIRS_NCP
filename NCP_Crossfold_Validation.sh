#!/bin/bash
#SBATCH -N 1
#SBATCH -n 10
#SBATCH --mem=5g
#SBATCH -J "Noise_Test - Alec Norton"
#SBATCH -p short
#SBATCH --gres=gpu:2
#SBATCH -C A100|V100
module load python/3.10.2
python3 -m venv myenv
source myenv/bin/activate
pip install tensorflow==2.15.0
pip install pandas
pip install numpy
pip install ncps
pip install glob
pip install sklearn
pip install matplotlib

module load cuda12.2

python NCP_Crossfold_Validation.py ${1} ${2} ${3} ${4} ${5} ${6} ${7} ${8}