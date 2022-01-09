#!/bin/bash

BASEDIR=`dirname $0`/..
pushd $BASEDIR/data
curl -L -o clausal-predictions.jsonl.gz https://www.dropbox.com/s/9inlepwnqddc70r/qasrl-qfirst-clausal-predictions.jsonl.gz?dl=1
gunzip clausal-predictions.jsonl.gz
popd