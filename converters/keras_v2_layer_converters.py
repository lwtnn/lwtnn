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
from keras_layer_converters_common import _activation_map

def _send_recieve_meta_info(backend):
    global BACKEND_SUFFIX
    BACKEND_SUFFIX = ":0" if backend == "tensorflow" else ""

def _get_dense_layer_parameters(h5, layer_config, n_in, *args):
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
        'activation': _activation_map[layer_config['activation']],
    }
    return return_dict, weights.shape[1]

def _normalization_parameters(h5, layer_config, n_in, *args):
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

def _rnn_parameters(h5, layer_config, n_in, *args):
    """RNN parameter converter. We support lstm and GRU """
    layer_group = h5[layer_config['name']]

    if "lstm" in args:
        elements = "ifco"
        rnn_architecure = "lstm"
    elif "gru" in args:
        elements = "zrh"
        rnn_architecure = "gru"
    else:
        sys.exit("We don't recognize the layer {}"
            .format(layer_config['name']))

    layers = _get_h5_layers(layer_group)
    n_out = layers['recurrent_kernel'+ BACKEND_SUFFIX].shape[0]
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

    return {'components': submap, 'architecture': rnn_architecure,
            'activation': _activation_map[layer_config['activation']],
            'inner_activation': _activation_map[layer_config['recurrent_activation']]}, n_out

def _get_merge_layer_parameters(h5, layer_config, n_in, *args):
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
        elif class_name == 'masking':
            dims = subcfg['batch_input_shape'][2]
            sum_inputs += dims
            sum_outputs += dims
        else:
            raise ValueError('unsupported merge layer {}'.format(class_name))

    assert sum_inputs == n_in
    return {'sublayers': sublayers, 'architecture': 'embedding',
            'activation': 'linear'}, sum_outputs


def _activation_parameters(h5, layer_config, n_in, *args):
    """Return dummy parameters"""
    return {'weights':[], 'bias':[], 'architecture':'dense',
            'activation':_activation_map[layer_config['activation']]}, n_in

# _________________________________________________________________________
# master list of converters

layer_converters = {
    'dense': _get_dense_layer_parameters,
    'batchnormalization': _normalization_parameters,
    'lstm': _rnn_parameters,
    'gru': _rnn_parameters,
    'merge': _get_merge_layer_parameters,
    'activation': _activation_parameters,
    }
# __________________________________________________________________________
# utilities

# utility function to handle keras layer naming
def _get_h5_layers(layer_group):
    """
    Keras: v2:
    An extra level is added, then the datasets we need. I.e., no stripping
    is needed, but one more iteration is.
    """
    layers = {}
    for long_name, ds in layer_group.items():
        for long_name1, ds1 in ds.items():
            layers[long_name1] = np.asarray(ds1)
    return layers

