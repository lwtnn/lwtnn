#!/usr/bin/env bash

# Coverage test exercising the ONNX ops not hit by the dense or batchnorm
# fixtures: Sigmoid, HardSigmoid, LeakyRelu and Elu with non-default
# alphas, plus MatMul+Add fusion and the full skip-op chain (Identity,
# Cast, Reshape, Squeeze, Unsqueeze, Dropout).

GENERATOR=./onnx/make_activations_onnx.py
MODEL=activations.onnx
VARIABLES=variables.json

CONVERT=./convert/onnx2json.py
TEST=./bin/lwtnn-test-lightweight-graph
OUTPUT=data/onnx-activations-out.json

echo " == Testing ONNX -> json -> C++ activations unit test ======="

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
