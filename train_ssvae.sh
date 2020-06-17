#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --job-name="mnist_ssvae_0.2"
#SBATCH --partition="msc"

export CONDA_ENVS_PATH=/scratch-ssd/$USER/conda_envs
export CONDA_PKGS_DIRS=/scratch-ssd/$USER/conda_pkgs

/scratch-ssd/oatml/scripts/run_locked.sh /scratch-ssd/oatml/miniconda3/bin/conda-env update -f gz_mizu.yml
echo 'env created'
source /scratch-ssd/oatml/miniconda3/bin/activate gz_mizu
echo 'env activated'
srun python train_ssvae.py --data_save tb_data_unsup_0.2
echo 'script ran'
