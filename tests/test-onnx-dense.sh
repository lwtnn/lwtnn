#!/usr/bin/env bash

# Unit test for the ONNX dense converter
#
# Full chain:
#  - Generate a small deterministic ONNX model + variables spec
#  - Convert to lwtnn JSON
#  - Run lwtnn-test-lightweight-graph on the JSON
#  - Compare outputs to the frozen reference

# _______________________________________________________________________
# define inputs, outputs, and code to run

# Fixture generator (creates dense.onnx and variables.json in a temp dir)
GENERATOR=./onnx/make_dense_onnx.py
MODEL=dense.onnx
VARIABLES=variables.json

# Conversion routine
CONVERT=./convert/onnx2json.py

# NN run routine
TEST=./bin/lwtnn-test-lightweight-graph

# Reference outputs
OUTPUT=data/onnx-dense-out.json

# Tell the tester what we're doing
echo " == Testing ONNX -> json -> C++ dense unit test ======="
echo " Consists of Gemm, Relu and Softmax layers"

# _______________________________________________________________________
# setup exit conditions and cleanup

set -eu

TMPDIR=$(mktemp -d)
echo "Will save temporary files to $TMPDIR"

function cleanup() {
    echo "cleaning up"
    rm -r $TMPDIR
}
trap cleanup EXIT

cd $(dirname ${BASH_SOURCE[0]})

# ________________________________________________________________________
# main test logic

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
