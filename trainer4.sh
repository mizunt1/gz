#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --job-name="ss_0.9999"
#SBATCH --partition="msc"

export CONDA_ENVS_PATH=/scratch-ssd/$USER/conda_envs
export CONDA_PKGS_DIRS=/scratch-ssd/$USER/conda_pkgs

/scratch-ssd/oatml/scripts/run_locked.sh /scratch-ssd/oatml/miniconda3/bin/conda-env update -f gz_mizu.yml
source /scratch-ssd/oatml/miniconda3/bin/activate gz_mizu

srun python trainer_vae_classifier_ss.py\
     --arch encoder_decoder_pose_vae_split.py\
     --csv_file /scratch-ssd/oatml/data/gz2/gz2_classifications_and_subjects.csv\
     --img_file /scratch-ssd/oatml/data/gz2\
     --dir_name trainer_classifier_ss_0.9999\
     --num_epochs 200 --img_size 128 --crop_size 128   --batch_size 100 \
     --lr 1e-4 --us_proportion 0.9999

