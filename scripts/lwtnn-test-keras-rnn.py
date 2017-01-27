#!/usr/bin/env python3

"""Run Keras as a compainion to lwtnn-test-rnn

With no `--test-inputs` argument, run with the standard "ramp"
function that `lwtnn-test-rnn` uses. When the test inputs
are given, expect two files:
 - The first should be a single line, one column for each input name
 - The second should be a single line, one column for each input value

"""
_help_arch_file = "NN archetecture file from Keras"
_help_vars_file = "Variable description file"
_help_hdf5_file = "NN weights file from Keras"
_help_n_in_sequence = (
    "number of patterns in test sequence (default: %(default)s)")

from argparse import ArgumentParser, RawDescriptionHelpFormatter
from numpy import linspace
import numpy as np
import json
from math import isnan
import os, sys

def _get_args():
    parser = ArgumentParser(
        description=__doc__,
        formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument('archetecture_file', help=_help_arch_file)
    parser.add_argument('variables_file', help=_help_vars_file)
    parser.add_argument('hdf5_file', help=_help_hdf5_file)
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument('-n', '--n-in-sequence', default=20,
                        type=int, help=_help_n_in_sequence)
    inputs.add_argument('-i','--test-inputs', nargs=2,
                        metavar=('NAMES', 'VALUES'))
    return parser.parse_args()

def run():
    args = _get_args()

    # keras loads slow, do the loading here
    from keras.models import model_from_json

    with open(args.archetecture_file) as arch:
        model = model_from_json(arch.read())
    model.load_weights(args.hdf5_file)

    with open(args.variables_file) as variables_file:
        inputs = json.loads(variables_file.read())

    n_in = model.layers[0].input_shape[2]
    assert n_in == len(inputs['inputs'])
    n_seq = args.n_in_sequence

    if args.test_inputs:
        test_pattern = _get_test_pattern(
            *args.test_inputs,
            input_dict=inputs['inputs'])
    else:
        test_pattern = _build_test_pattern(n_in, n_seq)

    outputs = list(model.predict(test_pattern))[0]
    out_pairs = sorted(zip(inputs['class_labels'], outputs))
    for name, val in out_pairs:
        print('{} {}'.format(name, val))

def _build_test_pattern(n_in, n_seq):
    input_vec = linspace(-1,1,n_in)[None,:]
    seq_vec = linspace(-1, 1, n_seq)[:,None]
    test_pattern = (input_vec * seq_vec)[None,...]
    return test_pattern

def _get_test_pattern(labels, values, input_dict):
    # todo: use this function in the feed-forward routine too
    with open(labels) as labels_file:
        field_keys = next(labels_file).split()

    value_transform = _get_value_transform(input_dict, field_keys)
    n_inputs = len(input_dict)
    assert n_inputs == len(field_keys)

    field_values = []
    with open(values) as values_file:
        for line in values_file:
            step_values = [float(x) for x in line.split()]
            assert len(field_keys) == len(step_values)
            field_values.append(step_values)

    normed_values = np.zeros((len(field_values), n_inputs))
    for step_n, step_v in enumerate(field_values):
        normed_values[step_n, :] = step_v
    return normed_values[None,:]

def _get_value_transform(inputs, field_keys):
    n_inputs = len(inputs)
    assert len(field_keys) == n_inputs
    pos_dict = {}
    scale = np.zeros((n_inputs,))
    offset = np.zeros((n_inputs,))
    for nnn, entry in enumerate(inputs):
        pos_dict[entry['name']] = nnn
        scale[nnn] = entry['scale']
        offset[nnn] = entry['offset']
    # build transform function
    def value_transform(field_values):
        raw_values = np.zeros((n_inputs,))
        for key, value in zip(field_keys, field_values):
            input_pos = pos_dict[key]
            if isnan(value):
                value = inputs[input_pos]['default']
            raw_values[input_pos] = value
        return (raw_values + offset) * scale
    return value_transform

if __name__ == '__main__':
    run()
