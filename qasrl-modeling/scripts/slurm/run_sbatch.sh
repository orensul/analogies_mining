#!/bin/bash

#SBATCH --job-name=qasrl-training
## Allocation
#SBATCH --account=cse
#SBATCH --partition=cse-gpu
## Output
#SBATCH --output=/gscratch/cse/julianjm/jobs/%j.out
#SBATCH --error=/gscratch/cse/julianjm/jobs/%j.err
## Nodes
#SBATCH --nodes=1
#SBATCH --tasks-per-node=4
#SBATCH --cpus-per-task=2
#SBATCH --signal=USR1@180
#SBATCH --open-mode=append
#SBATCH --time=16:00:00
#SBATCH --gres=gpu:4
#SBATCH --exclude=n2445

## Comment
###SBATCH --comment="ACL deadline Mar 4"

## training
srun --label /gscratch/cse/julianjm/qfirst/scripts/slurm_wrapper.sh
