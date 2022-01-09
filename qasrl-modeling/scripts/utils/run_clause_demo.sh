#!/bin/bash

BASE=`dirname $0`/..
pushd $BASE
{ echo ":load scripts/clause_demo.scala" & cat <&0; } | mill -i qfirst.clause-ext-demo.jvm.console
popd

