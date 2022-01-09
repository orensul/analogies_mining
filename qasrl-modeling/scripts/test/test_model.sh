#!/bin/bash
DIR=`dirname $1`
rm -rf $DIR/test
mkdir $DIR/test
if allennlp train $1 --include-package qasrl -s $DIR/test/save --overrides '{"train_data_path": "data/qasrl-dev-mini.jsonl", "validation_data_path": "data/qasrl-dev-mini.jsonl", "trainer": {"num_epochs": 1, "cuda_device": -1}}' 1> $DIR/test/stdout.log 2> $DIR/test/stderr.log ; then
  echo "Success: $1"
  rm -rf $DIR/test
else
  echo "FAILURE: $DIR/test/stderr.log"
fi
