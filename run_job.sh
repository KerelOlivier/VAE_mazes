#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH -p all
#SBATCH -N 1
#SBATCH --tasks-per-node 1
#SBATCH --output=R-%x.%j.out

source /home/TUE/20224456/miniconda3/etc/profile.d/conda.sh;
source activate torch_base;
echo "Training ConvVAE"
python main.py --config-path configs/cond_test/ConvVAE_config.yaml
echo "Training cConvVAE"
python main.py --config-path configs/cond_test/cConvVAE_config.yaml
echo "Training FcVAE"
python main.py --config-path configs/cond_test/FcVAE_config.yaml
echo "Training cFcVAE"
python main.py --config-path configs/cond_test/cFcVAE_config.yaml