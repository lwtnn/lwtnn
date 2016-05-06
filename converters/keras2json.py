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
    """
    For some reason Keras prefixes the datasets we need with the group
    name. This function returns a dictionary of the datasets, keyed
    with the group name stripped off.
    """
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
# Each of these converters takes three arguments:
#  - The H5 Group with the layer parameters
#  - The layer configuration
#  - The number of inputs (for error checking)
#
# Each returns two outputs:
#  - A dictionary of layer information which can be serialized to JSON
#  - The number of outputs (also for error checking)

def _get_dense_layer_parameters(h5, layer_config, n_in):
    """Get weights, bias, and n-outputs for a dense layer"""
    layer_group = h5[layer_config['name']]
    layers = _get_h5_layers(layer_group)
    weights = layers['W']
    bias = layers['b']
    assert weights.shape[1] == bias.shape[0]
    assert weights.shape[0] == n_in
    # TODO: confirm that we should be transposing the weight
    # matrix the Keras case
    return_dict = {
        'weights': weights.T.flatten('C').tolist(),
        'bias': bias.flatten('C').tolist(),
        'architecture': 'dense',
        'activation': layer_config['activation'],
    }
    return return_dict, weights.shape[1]


def _get_maxout_layer_parameters(h5, layer_config, n_in):
    """Get weights, bias, and n-outputs for a maxout layer"""
    layer_group = h5[layer_config['name']]
    layers = _get_h5_layers(layer_group)
    weights = layers['W']
    bias = layers['b']

    # checks (note the transposed arrays)
    wt_layers, wt_in, wt_out = weights.shape
    bias_layers, bias_n = bias.shape
    assert wt_out == bias_n
    assert wt_in == n_in, '{} != {}'.format(wt_in, n_in)
    assert wt_layers == bias_layers
    assert 'activation' not in layer_config

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
            'activation': 'linear'}, wt_out

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
    """
    Merge layer converter, currently only supports embedding, and only
    for the first layer.
    """
    sum_inputs = 0
    sum_outputs = 0
    sublayers = []
    for sublayer in layer_config['layers']:
        assert sublayer['class_name'].lower() == 'sequential'
        assert len(sublayer['config']) == 1
        subcfg = sublayer['config'][0]['config']
        class_name = sublayer['config'][0]['class_name'].lower()

        if class_name == 'embedding':
            layers = _get_h5_layers(h5[subcfg['name']])
            sublayer = {
                'weights': layers['W'].T.flatten().tolist(),
                'index': sum_inputs,
                'n_out': subcfg['output_dim']
                }
            sublayers.append(sublayer)
            sum_inputs += 1
            sum_outputs += subcfg['output_dim']
        elif class_name == 'activation':
            if subcfg['activation'] != 'linear':
                raise ValueError('we only support linear activation here')
            dims = subcfg['batch_input_shape'][2]
            sum_inputs += dims
            sum_outputs += dims
        else:
            raise ValueError('unsupported merge layer {}'.format(class_name))

    assert sum_inputs == n_in
    return {'sublayers': sublayers, 'architecture': 'embedding',
            'activation': 'linear'}, sum_outputs


def _activation_parameters(h5, layer_config, n_in):
    """Return dummy parameters"""
    return {'weights':[], 'bias':[], 'architecture':'dense',
            'activation':layer_config['activation']}, n_in

_layer_converters = {
    'dense': _get_dense_layer_parameters,
    'maxoutdense': _get_maxout_layer_parameters,
    'lstm': _lstm_parameters,
    'merge': _get_merge_layer_parameters,
    'activation': _activation_parameters,
    }
_skip_layers = {'flatten', 'dropout'}

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
        if layer_type in _skip_layers: continue
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
