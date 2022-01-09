#!/bin/bash
BASE=`dirname $0`
echo "Testing dummy models in $1/test..."
find $1/test -name config.json -exec $BASE/test_model.sh {} \;
echo "Testing real models in $1..."
find $1 -path */0.json -exec $BASE/test_model.sh {} \;
