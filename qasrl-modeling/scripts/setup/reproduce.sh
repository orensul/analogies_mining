#!/bin/bash

$CUDA_DEVICE=-1

### ANSWER-FIRST ###

$AFIRST_SPAN_DETECTOR_DIR=save/afirst/span_detector
$AFIRST_QUESTION_GENERATOR_DIR=save/afirst/question_generator
$AFIRST_PARSER_DIR=save/afirst

# train span detector
python -m allennlp.run train configs/afirst/span_detection.json --include-package qasrl -s $AFIRST_SPAN_DETECTOR_DIR
# train question generator
python -m allennlp.run train configs/afirst/question_generation.json --include-package qasrl -s $AFIRST_QUESTION_GENERATOR_DIR
# assemble into full model
mkdir -p $AFIRST_PARSER_DIR
python scripts/assemble_afirst_model.py \
       --config_base configs/afirst/parser_base.json \
       --span_detector $AFIRST_SPAN_DETECTOR_DIR \
       --question_generator $AFIRST_QUESTION_GENERATOR_DIR \
       --out $AFIRST_PARSER_DIR/model.tar.gz

# run predictions
mkdir -p $AFIRST_PARSER_DIR/predictions
allennlp predict \
         --predictor afirst \
         --cuda-device $CUDA_DEVICE \
         --include-package qasrl \
         --output-file $AFIRST_PARSER_DIR/predictions/predictions-dense.jsonl \
         $AFIRST_PARSER_DIR/model.tar.gz \
         qasrl-v2_1/dense/dev.jsonl

# TODO: run Scala evaluation code to get full results

### QUESTION-FIRST ###

$QFIRST_QUESTION_GENERATOR_DIR=save/qfirst/question_generator
$QFIRST_QUESTION_ANSWERER_DIR=save/qfirst/question_answerer
$QFIRST_PARSER_DIR=save/qfirst

# train question generator
python -m allennlp.run train configs/qfirst/question_generation.json --include-package qasrl -s $QFIRST_QUESTION_GENERATOR_DIR
# train question answerer
python -m allennlp.run train configs/qfirst/question_answering.json --include-package qasrl -s $QFIRST_QUESTION_ANSWERER_DIR
# assemble into full model
mkdir -p $QFIRST_PARSER_DIR
python scripts/assemble_afirst_model.py \
       --config_base configs/qfirst/parser_base.json \
       --question_generator $QFIRST_QUESTION_GENERATOR_DIR \
       --question_answerer $QFIRST_QUESTION_ANSWERER_DIR \
       --out $QFIRST_PARSER_DIR/model.tar.gz

# run predictions
mkdir -p $QFIRST_PARSER_DIR/predictions
allennlp predict \
         --predictor qfirst \
         --cuda-device $CUDA_DEVICE \
         --include-package qasrl \
         --output-file $QFIRST_PARSER_DIR/predictions/predictions-dense.jsonl \
         $QFIRST_PARSER_DIR/model.tar.gz \
         qasrl-v2_1/dense/dev.jsonl
