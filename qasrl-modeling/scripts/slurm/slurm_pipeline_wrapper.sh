#!/bin/bash

PATH=$PATH source /public/apps/anaconda3/5.0.1/bin/activate /private/home/jmichael/qfirst/env

export CUDA_VISIBLE_DEVICES=0

# PIPELINE="qfirst" or afirst or factored
# PARTITION="dev" or test

/private/home/jmichael/qfirst/env/bin/python \
    /private/home/jmichael/qfirst/qfirst/pipelines/${PIPELINE}_pipeline.py \
    --cuda_device 0 \
    --input_file /private/home/jmichael/qfirst/qasrl-v2_1/dense/${PARTITION}.jsonl.gz \
    "$@"
