#!/usr/bin/env bash

set -e
set -u
set -o pipefail

# Run all the tests
./tests/test-highway.sh
./tests/test-GRU.sh
./tests/test-BatchNorm.sh
./tests/test-dense_dropout_functional.sh
./tests/test-lstm_functional.sh
./tests/test-merge-graph.sh
./tests/test-time-distributed-dense.sh
./tests/test-gru-sequence.sh
./tests/check-version-number.sh
./tests/check-conversion.sh
./tests/test-leaky-relu.sh
./tests/test-elu.sh
