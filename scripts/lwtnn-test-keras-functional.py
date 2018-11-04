#!/usr/bin/env python3

"""
Run Keras as a compainion to lwtnn-test-lightweight-graph
"""
_help_arch_file = "NN archetecture file from Keras"
_help_vars_file = "Variable description file"
_help_hdf5_file = "Weights file from Keras"

from argparse import ArgumentParser, RawDescriptionHelpFormatter
from numpy import linspace
import numpy as np
import json
from CustomLayers import SwishBeta
from keras.utils.generic_utils import get_custom_objects
get_custom_objects().update({'SwishBeta': SwishBeta})

def _get_args():
    parser = ArgumentParser(
        description=__doc__,
        formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument('archetecture_file', help=_help_arch_file)
    parser.add_argument('variables_file', help=_help_vars_file)
    parser.add_argument('hdf5_file', help=_help_hdf5_file)
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

    full_test_pattern = []
    for in_node, in_spec in zip(model.inputs, inputs['inputs']):
        n_inputs = in_node.shape[1]
        assert n_inputs == len(in_spec['variables'])
        test_pattern = linspace(-1,1,n_inputs)[None,:]
        full_test_pattern.append(test_pattern)

    outputs = model.predict(full_test_pattern)
    for out_node, out_spec in zip(outputs, inputs['outputs']):
        out_pairs = sorted(zip(out_spec['labels'], out_node))
        print('{}:'.format(out_spec['name']))
        for name, val in out_pairs:
            print('{} {}'.format(name, val))

if __name__ == '__main__':
    run()
