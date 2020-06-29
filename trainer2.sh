#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --job-name="bar_nobar"
#SBATCH --partition="msc"

export CONDA_ENVS_PATH=/scratch-ssd/$USER/conda_envs
export CONDA_PKGS_DIRS=/scratch-ssd/$USER/conda_pkgs

/scratch-ssd/oatml/scripts/run_locked.sh /scratch-ssd/oatml/miniconda3/bin/conda-env update -f gz_mizu.yml
source /scratch-ssd/oatml/miniconda3/bin/activate gz_mizu

srun python trainer_vae_classifier_bar.py\
    --csv_file /scratch-ssd/oatml/data/gz2/gz2_classifications_and_subjects.csv\
    --img_file /scratch-ssd/oatml/data/gz2\
    --dir_name bar_no_bar2\
    --arch encoder_decoder_new_res3.py\
    --num_epochs 2700 --img_size 80 --crop_size 80 --z_size 100  --batch_size 100\
    --load_checkpoint classifier_test_fin
