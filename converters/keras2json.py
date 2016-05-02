#!/usr/bin/env python3
#
# Converter from Keras (version 1.0.0) saved NN to JSON
"""
____________________________________________________________________
Variable specification file

In additon to the standard Keras architecture and weights files, you
must provide a "variable specification" json file with the following
format:

  {
    "inputs": [
      {"name": variable_name,
       "scale": scale,
       "offset": offset,
       "default": default_value},
      ...
      ],
    "class_labels": [output_class_1_name, output_class_2_name, ...],
    "keras_version": "1.0.0"
  }

where `scale` and `offset` account for any scaling and shifting to the
input variables in preprocessing. The "default" value is optional.

"""

import argparse
import warnings
import json
import h5py
import numpy as np
from collections import Counter

def _run():
    """Top level routine"""
    args = _get_args()
    with open(args.arch_file, 'r') as arch_file:
        arch = json.load(arch_file)
    with open(args.variables_file, 'r') as inputs_file:
        inputs = json.load(inputs_file)

    if  inputs.get('keras_version')!="1.0.0":
        warnings.warn("This converter was developed for Keras version 1.0.0. \
        The provided files were generated using version {} and therefore \
        the conversion might break.".format(inputs.get('Keras version')))

    with h5py.File(args.hdf5_file, 'r') as h5:
        out_dict = {
            'layers': _get_layers(arch, inputs, h5),
        }
        out_dict.update(_parse_inputs(inputs))
    print(json.dumps(out_dict, indent=2))

def _get_args():
    parser = argparse.ArgumentParser(
        description="Converter from Keras saved NN to JSON",
        epilog=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('arch_file', help='architecture json file')
    parser.add_argument('variables_file', help='variable spec as json')
    parser.add_argument('hdf5_file', help='Keras weights file')
    return parser.parse_args()

# translate from keras to json representation
_activation_map = {
    'relu': 'rectified',
    'sigmoid': 'sigmoid',
    None: 'linear',
    # TODO: pass through unknown types rather than defining them as
    # themselves?
    'linear': 'linear',
    'softmax': 'softmax',
    'tanh': 'tanh',
}

# utility function to handle keras layer naming
def _get_h5_layers(layer_group):
    prefixes = set()
    numbers = set()
    layers = {}
    for name, ds in layer_group.items():
        prefix, number, name = name.split('_', 2)
        prefixes.add(prefix)
        numbers.add(number)
        layers[name] = np.asarray(ds)
    assert len(prefixes) == 1, 'too many prefixes: {}'.format(prefixes)
    assert len(numbers) == 1, 'too many numbers: {}'.format(numbers)
    return layers

# __________________________________________________________________________
# Layer converters
#
# Each of these converters takes two arguments:
#  - An H5 Group with the layer parameters
#  - The number of inputs (for error checking)
#
# Each returns two outputs:
#  - A dictionary of layer information which can be serialized to JSON
#  - The number of outputs (also for error checking)


def _get_dense_layer_parameters(h5, layer_config, n_in):
    """Get weights, bias, and n-outputs for a dense layer"""
    layer_group = h5[layer_config['name']]
    weights = layer_group.get(list(layer_group.keys())[0])
    bias = layer_group.get(list(layer_group.keys())[1])
    assert weights.shape[1] == bias.shape[0]
    assert weights.shape[0] == n_in
    # TODO: confirm that we should be transposing the weight
    # matrix the Keras case
    return_dict = {
        'weights': np.asarray(weights).T.flatten('C').tolist(),
        'bias': np.asarray(bias).flatten('C').tolist(),
        'architecture': 'dense',
        'activation': layer_config['activation'],
    }
    return return_dict, weights.shape[1]


def _get_maxout_layer_parameters(h5, layer_config, n_in):
    """Get weights, bias, and n-outputs for a maxout layer"""
    layer_group = h5[layer_config['name']]
    # TODO: make this more robust, should look by key, not key index
    weights = np.asarray(layer_group.get(list(layer_group.keys())[0]))
    bias = np.asarray(layer_group.get(list(layer_group.keys())[1]))

    # checks (note the transposed arrays)
    wt_layers, wt_in, wt_out = weights.shape
    bias_layers, bias_n = bias.shape
    assert wt_out == bias_n
    assert wt_in == n_in, '{} != {}'.format(wt_in, n_in)
    assert wt_layers == bias_layers

    sublayers = []
    for nnn in range(weights.shape[0]):
        w_slice = weights[nnn,:,:]
        b_slice = bias[nnn,:]
        sublayer = {
            'weights': w_slice.T.flatten().tolist(),
            'bias': b_slice.flatten().tolist(),
            'architecture': 'dense'
        }
        sublayers.append(sublayer)
    return {'sublayers': sublayers, 'architecture': 'maxout',
            'activation': layer_config['activation']}, wt_out

def _lstm_parameters(h5, layer_config, n_in):
    """LSTM parameter converter"""
    layer_group = h5[layer_config['name']]
    layers = _get_h5_layers(layer_group)
    n_out = layers['W_o'].shape[1]

    submap = {}
    for gate in 'cfio':
        submap[gate] = {
            'U': layers['U_' + gate].T.flatten().tolist(),
            'weights': layers['W_' + gate].T.flatten().tolist(),
            'bias': layers['b_' + gate].flatten().tolist(),
        }
        # TODO: add activation function for some of these gates
    return {'components': submap, 'architecture': 'lstm',
            'activation': layer_config['activation'],
            'inner_activation': layer_config['inner_activation']}, n_out

def _get_merge_layer_parameters(h5, layer_config, n_in):
    """Merge layer converter, currently only supports embedding"""
    # layer_group = h5[layer_config['name']]
    layers = _get_h5_layers(layer_group)


def _dummy_parameters(h5, layer_config, n_in):
    """Return dummy parameters"""
    return {'weights':[], 'bias':[], 'architecture':'dense',
            'activation':layer_config['activation']}, n_in

_layer_converters = {
    'dense': _get_dense_layer_parameters,
    'maxoutdense': _get_maxout_layer_parameters,
    'lstm': _lstm_parameters,
    'merge': _get_merge_layer_parameters,
    'activation': _dummy_parameters,
    'flatten': _dummy_parameters,
    }

# __________________________________________________________________________
# master layer converter / inputs function

def _get_layers(network, inputs, h5):
    layers = []
    in_layers = network['config']
    n_out = len(inputs['inputs'])
    for layer_n in range(len(in_layers)):
        # get converter for this layer
        layer_arch = in_layers[layer_n]
        layer_type = layer_arch['class_name'].lower()
        convert = _layer_converters[layer_type]

        # build the out layer
        out_layer, n_out = convert(h5, layer_arch['config'], n_out)
        layers.append(out_layer)
    return layers

def _parse_inputs(keras_dict):
    inputs = []
    defaults = {}
    for val in keras_dict['inputs']:
        inputs.append({x: val[x] for x in ['offset', 'scale', 'name']})

        # maybe fill default
        default = val.get("default")
        if default is not None:
            defaults[val['name']] = default
    return {
        'inputs': inputs,
        'outputs': keras_dict['class_labels'],
        'defaults': defaults,
    }

if __name__ == '__main__':
    _run()
