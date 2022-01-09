
#!/bin/bash

#SBATCH --job-name=qasrl-tuning
#SBATCH --output=/checkpoint/%u/jobs/%j.out
#SBATCH --error=/checkpoint/%u/jobs/%j.err
#SBATCH --partition=dev
#SBATCH --cpus-per-task=2
#SBATCH --signal=USR1@180
#SBATCH --open-mode=append
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:volta:1
###SBATCH --comment="ACL deadline Mar 4"

# Predicting
# Needs env vars PIPELINE and PARTITION
module load anaconda3
srun /private/home/jmichael/qfirst/scripts/slurm_pipeline_wrapper.sh "$@"
