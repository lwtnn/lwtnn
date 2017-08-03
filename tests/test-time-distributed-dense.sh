#!/usr/bin/env bash

# Example unit test for lwtnn code
#
# This should run the full chain:
#  - Convert a saved NN to JSON
#  - Read in the saved file, with C++, test on a few patterns
#  - Compare the outputs to somehting previously saved
#
# If feel free to copy and modify this file, and add tests to it to
# the `.travis.yml` file.

# _______________________________________________________________________
# define inputs, outputs, and code to run
#
# If to add another test you'll probably have to edit this

# Trained network to convert and test
TARBALL=time-distributed.tgz
INPUT=https://github.com/lwtnn/lwtnn-test-data/raw/v4/${TARBALL}
if [[ ${LWTNN_LOCAL_TEST_DATA+x} ]] ; then
    INPUT=${LWTNN_LOCAL_TEST_DATA}/${TARBALL}
fi
ARCH=tdd-architecture.json
VARIABLES=tdd-inputs.json
HDF5=tdd-weights.h5

# Conversion routine
CONVERT=./convert/kerasfunc2json.py

# NN run routine, which loads in the network, and writes to stdout
TEST=./bin/lwtnn-test-lightweight-graph

# Target file, can be generated by piping the NN output through the
# `reg-test.py` script. We also use the `reg-test.py` routine to make
# sure outputs haven't changed. See `reg-test.py -h` for more info.
OUTPUT=tdd-targets.json

# Tell the tester what we're doing
echo " === Keras -> json -> C++ time-distributed unit test ======="
echo " Consists of time distributed dense and gru layer"

# _______________________________________________________________________
# setup exit conditions and cleanup
# (you probably don't have to touch this)

# exit with any nonzero return codes
set -eu

# build a temp dir to store any intermediate
TMPDIR=$(mktemp -d)
echo "Will save temporary files to $TMPDIR"

# cleanup function gets called on exit
function cleanup() {
    echo "cleaning up"
    rm -r $TMPDIR
}
trap cleanup EXIT

# go to the directory where the script lives
cd $(dirname ${BASH_SOURCE[0]})

# ________________________________________________________________________
# main test logic

# make sure the inputs / outputs are there
#
# If you're adding another test other tests, I'd recommend downloading
# or otherwise acquiring the input files by running `wget` within this
# block. Make sure you set the input path outside the block, it will
# go out of scope otherwise.
(
    cd $TMPDIR
    # get the data here!
    # for example:
    echo " -- downloading and unpacking data --"
    if [[ ${LWTNN_LOCAL_TEST_DATA+x} ]] ; then
        cp ${INPUT} .
    else
        wget -nv $INPUT
    fi
    tar xf ${INPUT##*/}
    if [[ ! -f $ARCH || ! -f $HDF5 || ! -f $VARIABLES ]] ; then
        echo "missing some inputs to the keras -> json converter" >&2
        exit 1
    fi
    if [[ ! -f $OUTPUT ]]; then
        echo "no output found" >&2
        exit 1
    fi
)

# intermediate file name (make sure it's in the temp dir)
JSON_FILE=$TMPDIR/intermediate.json

# run the conversion
echo " -- Running conversion $CONVERT $ARCH $HDF5 $VARIABLES --"
$CONVERT $TMPDIR/$ARCH $TMPDIR/$HDF5 $TMPDIR/$VARIABLES  > $JSON_FILE
# check that it hasn't changed!
echo "Testing with $TEST"
$TEST $JSON_FILE | ./reg-test.py --graph $TMPDIR/$OUTPUT

echo " *** Success! ***"
