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

import numpy as np
import sys
from keras_layer_converters_common import activation_map
import h5py

BACKEND_SUFFIX = ''

def _get_dense_layer_parameters(h5, layer_config, n_in, layer_type):
    """Get weights, bias, and n-outputs for a dense layer"""
    layer_group = h5[layer_config['name']]
    layers = _get_h5_layers(layer_group)
    weights = layers["kernel"+BACKEND_SUFFIX]
    bias = layers["bias"+BACKEND_SUFFIX]
    assert weights.shape[1] == bias.shape[0]
    assert weights.shape[0] == n_in
    # TODO: confirm that we should be transposing the weight
    # matrix the Keras case
    return_dict = {
        'weights': weights.T.flatten('C').tolist(),
        'bias': bias.flatten('C').tolist(),
        'architecture': 'dense',
        'activation': activation_map[layer_config['activation']],
    }
    return return_dict, weights.shape[1]

def _time_distributed_parameters(h5, layer_config, n_in, layer_type):
    dist_layer = layer_config['layer']['config']
    # TODO: Come up with a cleaner way to pass the h5 group
    # corresponding to the wrapped layer into the layer converter. To
    # do this we need to figure out exactly how Keras uses the json
    # archetecture to locate h5 datasets.
    dist_layer['name'] = layer_config['name']
    dist_name = layer_config['layer']['class_name'].lower()
    return layer_converters[dist_name](h5, dist_layer, n_in, layer_type)

def _normalization_parameters(h5, layer_config, n_in, layer_type):
    """Get weights (gamma), bias (beta), for normalization layer"""
    layer_group = h5[layer_config['name']]
    layers = _get_h5_layers(layer_group)
    gamma = layers['gamma'+BACKEND_SUFFIX]
    beta = layers['beta'+BACKEND_SUFFIX]
    mean = layers['moving_mean'+BACKEND_SUFFIX]
    stddev = layers['moving_variance'+BACKEND_SUFFIX]
    # Do some checks
    assert gamma.shape[0] == beta.shape[0]
    assert mean.shape[0] == stddev.shape[0]
    assert gamma.shape[0] == n_in
    assert 'activation' not in layer_config
    epsilon = layer_config['epsilon']
    scale = gamma / np.sqrt(stddev + epsilon)
    offset = -mean+(beta*np.sqrt(stddev + epsilon)/(gamma))
    return_dict = {
        'weights': scale.T.flatten('C').tolist(),
        'bias': offset.flatten('C').tolist(),
        'architecture': 'normalization',
    }
    return return_dict, scale.shape[0]

def _rnn_parameters(h5, layer_config, n_in, layer_type):
    """RNN parameter converter. We support lstm and GRU """
    layer_group = h5[layer_config['name']]

    if "lstm" in layer_type:
        elements = "ifco"
        rnn_architecture = "lstm"
    elif "gru" in layer_type:
        elements = "zrh"
        rnn_architecture = "gru"
    elif "simplernn" in layer_type:
        elements = "h"
        rnn_architecture = "simplernn"
    else:
        sys.exit("We don't recognize the layer {}"
                 .format(layer_type))

    layers = _get_h5_layers(layer_group)
    n_out = layers['recurrent_kernel' + BACKEND_SUFFIX].shape[0]
    submap = {}

    for n_gate, gate in enumerate(elements):
        submap[gate] = {
            'U': layers['recurrent_kernel'+BACKEND_SUFFIX]\
                [:, n_out*n_gate : n_out*(1+n_gate)].T.flatten().tolist(),
            'weights': layers['kernel'+BACKEND_SUFFIX]\
                [:, n_out*n_gate : n_out*(1+n_gate)].T.flatten().tolist(),
            'bias': layers['bias'+BACKEND_SUFFIX]\
                [n_out*n_gate : n_out*(1+n_gate)].flatten().tolist(),
        }

    rnn_parameters = {'components': submap, 'architecture': rnn_architecture,
                      'activation': activation_map[layer_config['activation']]}

    if rnn_architecture != "simplernn":
        rnn_parameters['inner_activation'] = activation_map[layer_config['recurrent_activation']]


    return rnn_parameters, n_out

def _conv1d_parameters(h5, layer_config, n_in, layer_type):
    """CNN parameter converter. We only support 1D kernels."""
    layer_group = h5[layer_config['name']]
    layers = _get_h5_layers(layer_group)
    weights = layers["kernel"+BACKEND_SUFFIX]
    bias = layers["bias"+BACKEND_SUFFIX]
    assert weights.shape[-1] == bias.shape[0]
    assert weights.shape[1] == n_in
    return_dict = {
        'weights': weights.T.flatten('C').tolist(),
        'dilation_rate': layer_config['dilation_rate'][0],
        'padding': layer_config['padding'],
        'bias': bias.flatten('C').tolist(),
        'architecture': 'conv1d',
        'activation': activation_map[layer_config['activation']]
    }
    return return_dict, weights.shape[-1]

def _activation_parameters(h5, layer_config, n_in, layer_type):
    """Return dummy parameters"""
    return {'weights':[], 'bias':[], 'architecture':'dense',
            'activation':activation_map[layer_config['activation']]}, n_in

def _activation_func(activation_name):
    def func(h5, layer_config, n_in, layer_type):
        """Return dummy parameters"""
        return {'weights':[], 'bias':[], 'architecture':'dense',
                'activation': activation_name}, n_in
    return func

def _alpha_activation_func(activation_name):
    def func(h5, layer_config, n_in, layer_type):
        """Store activation parameters, including alpha"""
        pars = {
            'weights':[], 'bias':[],'architecture':'dense',
            'activation':{
                'function': activation_name,
                'alpha': layer_config['alpha']
            }
        }
        return pars, n_in
    return func

def _trainable_alpha_activation_function(activation_name, alpha_parameter_name='alpha'):
    def func(h5, layer_config, n_in, layer_type):
        """Store single trainable activation parameter"""
        pars = {
            'weights':[], 'bias':[],'architecture':'dense',
            'activation':{
                'function': activation_name,
                'alpha': layer_config[alpha_parameter_name],

            }
        }
        return pars, n_in
    return func


# _________________________________________________________________________
# master list of converters

layer_converters = {
    'dense': _get_dense_layer_parameters,
    'batchnormalization': _normalization_parameters,
    'lstm': _rnn_parameters,
    'gru': _rnn_parameters,
    'simplernn': _rnn_parameters,
    'conv1d': _conv1d_parameters,
    'activation': _activation_parameters,
    'softmax': _activation_func('softmax'),
    'leakyrelu': _alpha_activation_func('leakyrelu'),
    'swish': _trainable_alpha_activation_function('swish', alpha_parameter_name='beta'),
    'timedistributed': _time_distributed_parameters,
    }
# __________________________________________________________________________
# utilities

# utility function to handle keras layer naming
def _get_h5_layers(layer_group):
    """
    Keras: v2:
    Recursively loop over the groups until a dataset is found
    """
    for long_name, ds in layer_group.items():
        layer_info = _get_h5_layers_recursively(ds)
    return layer_info


def _get_h5_layers_recursively(dataset):
        layers={}
        for long_name1, ds1 in dataset.items():
            is_dataset = isinstance(ds1, h5py.Dataset)
            if is_dataset==True:
                layers[long_name1] = np.asarray(ds1)
            else:
                return _get_h5_layers_recursively(ds1)
        return layers
