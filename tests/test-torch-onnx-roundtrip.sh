#!/usr/bin/env bash

# torch -> ONNX -> lwtnn roundtrip test.
#
# Builds a small MLP in PyTorch, exports it to ONNX, converts to lwtnn JSON,
# and verifies that lwtnn's output matches the torch forward pass on a
# fixed input within reg-test.py's tolerance.
#
# Skipped (success) if PyTorch is not installed in the test environment.

GENERATOR=./onnx/make_torch_roundtrip.py
MODEL=model.onnx
VARIABLES=variables.json
INPUTS=inputs.json
EXPECTED=expected.json

CONVERT=./convert/onnx2json.py
TEST=./bin/lwtnn-test-lightweight-graph

echo " == Testing torch -> ONNX -> json -> C++ roundtrip ======="

set -eu

cd $(dirname ${BASH_SOURCE[0]})

if ! python -c 'import torch' 2>/dev/null; then
    echo " -- PyTorch not available, skipping torch roundtrip test --"
    exit 0
fi

TMPDIR=$(mktemp -d)
echo "Will save temporary files to $TMPDIR"

function cleanup() {
    echo "cleaning up"
    rm -r $TMPDIR
}
trap cleanup EXIT

echo " -- Building torch model and exporting to ONNX --"
python $GENERATOR $TMPDIR
for f in $MODEL $VARIABLES $INPUTS $EXPECTED; do
    if [[ ! -f $TMPDIR/$f ]]; then
        echo "fixture generator did not produce $f" >&2
        exit 1
    fi
done

JSON_FILE=$TMPDIR/intermediate.json

echo " -- Running conversion $CONVERT $MODEL $VARIABLES --"
$CONVERT $TMPDIR/$MODEL $TMPDIR/$VARIABLES > $JSON_FILE

echo "Testing with $TEST against torch-frozen expected output"
$TEST $JSON_FILE $TMPDIR/$INPUTS | ./reg-test.py --graph $TMPDIR/$EXPECTED

echo " *** Success! ***"
