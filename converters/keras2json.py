#!/usr/bin/env python3
"""
Converter from Keras saved NN to JSON
"""

# import yaml
import argparse
import json
import h5py
import numpy as np

def _run():
    """Top level routine"""
    args = _get_args()
    with open(args.arch_file, 'r') as arch_file:
        arch = json.load(arch_file)
    with open(args.inputs_file, 'r') as inputs_file:
        inputs = json.load(inputs_file)
    with h5py.File(args.hdf5_file, 'r') as h5:
        out_dict = {
            'layers': _get_layers(arch, h5),
        }
        out_dict.update(_parse_inputs(inputs))
    print(json.dumps(out_dict, indent=2))

def _get_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('arch_file', help='architecture json file')
    parser.add_argument('inputs_file', help='input specifications json')
    parser.add_argument('hdf5_file')
    return parser.parse_args()

_activation_map = {
    'relu': 'rectified',
    'sigmoid': 'sigmoid',
}

def _get_layers(network, h5):
    layers = []
    in_layers = network['layers']
    n_out = h5['layer_0']['param_0'].shape[0]
    for layer_n in range(len(in_layers)):
        activation = _activation_map[in_layers[layer_n]['activation']]
        layer_group = h5['layer_{}'.format(layer_n)]
        weights = layer_group['param_0']
        bias = layer_group['param_1']
        assert weights.shape[1] == bias.shape[0]
        assert weights.shape[0] == n_out
        n_out = weights.shape[1]
        out_layer = {
            'activation': activation,
            'weights': np.asarray(weights).flatten('F').tolist(),
            'bias': np.asarray(bias).flatten('F').tolist()
        }
        layers.append(out_layer)
    return layers

def _parse_inputs(keras_dict):
    # fill output names
    keras_outputs = keras_dict['individual_class_info']
    outputs = [None]*len(keras_outputs)
    for key, val in keras_outputs.items():
        outputs[val] = key
    assert all(x is not None for x in outputs)

    keras_inputs = keras_dict['inputs']
    inputs = [None]*len(keras_inputs)
    defaults = {}
    # fill the other things
    for input_name, val in keras_inputs.items():
        number = val["input_number"]
        inputs[number] = {
            'name': input_name,
            'offset': val["offset"],
            'scale': val["scale"],
        }

        # maybe fill default
        default = val.get("default")
        if default is not None:
            defaults[input_name] = default
    return {
        'inputs': inputs,
        'outputs': outputs,
        'defaults': defaults,
    }

if __name__ == '__main__':
    _run()
