#!/bin/bash
#PATH=$PATH source /gscratch/cse/julianjm/anaconda3/bin/activate /private/home/jmichael/qfirst/env
export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
MODEL_VARIANT=$(($INDEX_OFFSET + $SLURM_LOCALID))
THIS_MODEL_BRANCH=$MODELS_BRANCH/$MODEL_VARIANT.json
echo $SLURMD_NODENAME $SLURM_JOB_ID $SLURM_LOCALID $THIS_MODEL_BRANCH $INIT_BATCH_SIZE
echo `which python`
python /gscratch/cse/julianjm/qfirst/qfirst/training/run_slurm.py \
  $MODELS_ROOT $THIS_MODEL_BRANCH $INIT_BATCH_SIZE
