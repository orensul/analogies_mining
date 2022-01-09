#!/bin/bash

git submodule update --init

# create project-local venv
python3.6 -m venv env
# install python dependencies into local venv
source env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# download pretrained QA-SRL model
pushd lib/nrl-qasrl/scripts
./download_pretrained.sh
popd

mv lib/nrl-qasrl/data data

# fix up config of pretrained model
pushd data/qasrl_parser_elmo
sed -i -e 's/data\/elmo/https:\/\/s3-us-west-2.amazonaws.com\/allennlp\/models\/elmo\/2x4096_512_2048cnn_2xhighway/g' config.json
sed -i -e 's/\/home\/nfitz\/data\/qasrl-v2/http:\/\/qasrl.org\/data\/qasrl-v2/g' config.json
sed -i -e 's/data\/glove/https:\/\/s3-us-west-2.amazonaws.com\/allennlp\/datasets\/glove/g' config.json
# TODO remove "type" labels and from config as well as "hidden_dim" from question predictor
popd


pushd lib/nrl-qasrl
sed -i -e 's/data\/glove/https:\/\/s3-us-west-2.amazonaws.com\/allennlp\/datasets\/glove/g' nrl/service/predictors/qasrl_parser.py
popd

deactivate
