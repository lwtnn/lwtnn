#!/usr/bin/env bash

# Unit test for the ONNX converter, exercising BatchNormalization, Concat
# and graph-level Add merge.

GENERATOR=./onnx/make_batchnorm_onnx.py
MODEL=bn.onnx
VARIABLES=variables.json

CONVERT=./convert/onnx2json.py
TEST=./bin/lwtnn-test-lightweight-graph
OUTPUT=data/onnx-batchnorm-out.json

echo " == Testing ONNX -> json -> C++ BatchNorm unit test ======="
echo " Consists of Gemm, BatchNormalization, Relu, Concat and Add layers"

set -eu

TMPDIR=$(mktemp -d)
echo "Will save temporary files to $TMPDIR"

function cleanup() {
    echo "cleaning up"
    rm -r $TMPDIR
}
trap cleanup EXIT

cd $(dirname ${BASH_SOURCE[0]})

if [[ ! -f $OUTPUT ]]; then
    echo "no output found" >&2
    exit 1
fi

echo " -- Generating fixture --"
python $GENERATOR $TMPDIR
if [[ ! -f $TMPDIR/$MODEL || ! -f $TMPDIR/$VARIABLES ]]; then
    echo "fixture generator did not produce expected files" >&2
    exit 1
fi

JSON_FILE=$TMPDIR/intermediate.json

echo " -- Running conversion $CONVERT $MODEL $VARIABLES --"
$CONVERT $TMPDIR/$MODEL $TMPDIR/$VARIABLES > $JSON_FILE

echo "Testing with $TEST"
$TEST $JSON_FILE | ./reg-test.py --graph $OUTPUT

echo " *** Success! ***"
