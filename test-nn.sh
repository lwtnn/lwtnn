#!/usr/bin/env bash

# the `agile2json.py` converter will write a json file to stdout
# we pipe this into `test-full`, which expects the json file in cin.
./converters/agile2json.py data/test-nn.yml | ./bin/lwtnn-test-full
