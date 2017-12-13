#!/usr/bin/env bash

# Example unit test for lwtnn converter code
#
# This should:
#  - Create an inputs file and compare it to a template
#  - Create a network file and compare it to a template
#
# If feel free to copy and modify this file, and add tests to it to
# the `.travis.yml` file.

# _______________________________________________________________________
# define inputs, outputs, and code to run
#
# If to add another test you'll probably have to edit this

# Trained network to convert and test
INPUT=https://github.com/lwtnn/lwtnn-test-data/raw/v3/lstm_functional.tgz
ARCH=model.json
VARIABLES_ORIGINAL=inputs.json
HDF5=weights.h5
#Correct file
JSON_FILE_OLD=data/converter_data/basic_lstm_network_output.json

# Conversion routine
CONVERT=./convert/kerasfunc2json.py

# Tell the tester what we're doing
echo " == Keras -> input json -> network output json ======="
echo " Checking conversion process "

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

# make sure the NN to compare to is there
if [[ ! -f $JSON_FILE_OLD ]]; then
    echo "no comparison NN found" >&2
    exit 1
fi

# Function to check if file is empty (as it should be) or not
function getFileSize(){
  if [[ -s $1 ]];
  then
    echo "ERROR!  Diff file is NOT empty"
  else
    echo "Success! Diff file is empty";
  fi;
}


# If you're adding another test other tests, I'd recommend downloading
# or otherwise acquiring the input files by running `wget` within this
# block. Make sure you set the input path outside the block, it will
# go out of scope otherwise.
(
    cd $TMPDIR
    # get the data here!
    # for example:
    echo " -- downloading and unpacking data --"
    wget -nv $INPUT
    tar xf ${INPUT##*/}
    if [[ ! -f $ARCH || ! -f $HDF5 || ! -f $VARIABLES_ORIGINAL ]] ; then
        echo "missing some inputs to the keras -> json converter" >&2
        exit 1
    fi
)

# intermediate file name (make sure it's in the temp dir)
JSON_FILE_NEW=$TMPDIR/intermediate_model.json
INPUT_FILE_NEW=$TMPDIR/intermediate_inputs.json

# run the conversions
echo " -- Creating inputs $CONVERT $ARCH $HDF5 > $INPUT_FILE_NEW --"
$CONVERT $TMPDIR/$ARCH $TMPDIR/$HDF5 > $INPUT_FILE_NEW
# Now compare the $INPUT_FILE_NEW to the $VARIABLES_ORIGINAL
diff -u $INPUT_FILE_NEW $TMPDIR/$VARIABLES_ORIGINAL > $TMPDIR/input_differences.txt
getFileSize $TMPDIR/input_differences.txt

echo " -- Creating networks $CONVERT $ARCH $HDF5 $INPUT_FILE_NEW > $JSON_FILE_NEW --"
$CONVERT $TMPDIR/$ARCH $TMPDIR/$HDF5 $INPUT_FILE_NEW  > $JSON_FILE_NEW
# Now compare the $JSON_FILE_NEW to the $JSON_FILE_OLD
diff -u $JSON_FILE_NEW $JSON_FILE_OLD  > $TMPDIR/neural_network_differences.txt
getFileSize $TMPDIR/neural_network_differences.txt

echo " *** Success! ***"
