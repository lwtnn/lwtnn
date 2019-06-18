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
from CustomLayers import Swish, Sum
from keras.utils.generic_utils import get_custom_objects
get_custom_objects().update({'Swish': Swish, 'Sum': Sum})

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
    vec_input_iterator = iter(inputs['inputs'])
    seq_input_iterator = iter(inputs['input_sequences'])
    for in_node in model.inputs:
        print(in_node.shape)
        if len(in_node.shape) == 2:
            in_spec = next(vec_input_iterator)
            n_inputs = in_node.shape[1]
            test_pattern = linspace(-1,1,n_inputs)[None,:]
        elif len(in_node.shape) == 3:
            in_spec = next(seq_input_iterator)
            n_inputs = in_node.shape[2]
            input_vec = linspace(-1,1,n_inputs)[None,:]
            seq_vec = linspace(-1, 1, 20)[:,None]
            test_pattern = (input_vec * seq_vec)[None,...]
        else:
            raise RuntimeError(
                "not sure what to do with input sequence lentgh {}".format(
                    len(in_node.shape)))
        if n_inputs != len(in_spec['variables']):
            var_names = [v['name'] for v in in_spec['variables']]
            raise RuntimeError(
                "need {} inputs for variables: {}".format(
                    n_inputs, ', '.join(var_names)))

        full_test_pattern.append(test_pattern)

    outputs = model.predict(full_test_pattern)
    for out_node, out_spec in zip(outputs, inputs['outputs']):
        out_pairs = sorted(zip(out_spec['labels'], out_node))
        print('{}:'.format(out_spec['name']))
        for name, val in out_pairs:
            print('{} {}'.format(name, val))

if __name__ == '__main__':
    run()
