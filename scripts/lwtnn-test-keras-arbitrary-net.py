#!/usr/bin/env python3

"""Run Keras as a compainion to lwtnn-test-arbitrary-net

With no `--test-inputs` argument, run with the standard "ramp"
function that `lwtnn-test-arbitrary-net` uses. When the test inputs
are given, expect two files:
 - The first should be a single line, one column for each input name
 - The second should be a single line, one column for each input value

"""
_help_arch_file = "NN archetecture file from Keras"
_help_vars_file = "Variable description file"
_help_hdf5_file = "NN weights file from Keras"

from argparse import ArgumentParser, RawDescriptionHelpFormatter
from numpy import linspace
import numpy as np
import json
from math import isnan

def _get_args():
    parser = ArgumentParser(
        description=__doc__,
        formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument('archetecture_file', help=_help_arch_file)
    parser.add_argument('variables_file', help=_help_vars_file)
    parser.add_argument('hdf5_file', help=_help_hdf5_file)
    parser.add_argument('-i','--test-inputs', nargs=2,
                        metavar=('NAMES', 'VALUES'))
    return parser.parse_args()


def run():
    args = _get_args()

    # keras loads slow, do the loading here
    from keras.models import model_from_json

    with open(args.archetecture_file) as arch:
        model = model_from_json(''.join(arch.readlines()))
    model.load_weights(args.hdf5_file)

    with open(args.variables_file) as variables_file:
        inputs = json.loads(''.join(variables_file.readlines()))

    n_inputs = model.layers[0].input_shape[1]
    assert n_inputs == len(inputs['inputs'])

    if not args.test_inputs:
        test_pattern = linspace(-1,1,n_inputs)[None,:]
    else:
        test_pattern = _get_test_pattern(*args.test_inputs, input_dict=inputs)
    outputs = list(model.predict(test_pattern))[0]
    out_pairs = sorted(zip(inputs['class_labels'], outputs))
    for name, val in out_pairs:
        print('{} {}'.format(name, val))

def _get_test_pattern(labels, values, input_dict):
    with open(labels) as labels_file:
        field_keys = next(labels_file).split()
    with open(values) as values_file:
        field_values = [float(x) for x in next(values_file).split()]

    n_inputs = len(field_values)

    assert len(field_keys) == len(field_values)
    assert n_inputs == len(input_dict['inputs'])

    pos_dict = {}
    scale = np.zeros((n_inputs,))
    offset = np.zeros((n_inputs,))
    for nnn, entry in enumerate(input_dict['inputs']):
        pos_dict[entry['name']] = nnn
        scale[nnn] = entry['scale']
        offset[nnn] = entry['offset']

    raw_values = np.zeros((n_inputs,))
    for key, value in zip(field_keys, field_values):
        input_pos = pos_dict[key]
        if isnan(value):
            value = input_dict['inputs'][input_pos]['default']
        raw_values[input_pos] = value

    normed_values = (raw_values + offset) * scale
    return normed_values[None,:]

if __name__ == '__main__':
    run()
