#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --job-name="gz_vae"
#SBATCH --partition="msc"

export CONDA_ENVS_PATH=/scratch-ssd/$USER/conda_envs
export CONDA_PKGS_DIRS=/scratch-ssd/$USER/conda_pkgs

/scratch-ssd/oatml/scripts/run_locked.sh /scratch-ssd/oatml/miniconda3/bin/conda-env update -f my_env.yml
source /scratch-ssd/oatml/miniconda3/bin/activate my_env.yml

srun python gz_vae_train.py\
     --csv_file /scratch-ssd/oatml/data/gz2/gz2_classifications_and_subjects.csv\
     --img_file /scratch-ssd/oatml/data/gz2\
     --use_cuda True
