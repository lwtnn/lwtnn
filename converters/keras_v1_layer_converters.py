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
from keras_layer_converters_common import _activation_map

def _send_recieve_meta_info(backend):
    global BACKEND_SUFFIX
    BACKEND_SUFFIX = ":0" if backend == "tensorflow" else ""

def _get_dense_layer_parameters(h5, layer_config, n_in, layer_type):
    """Get weights, bias, and n-outputs for a dense layer"""
    if layer_type in ['timedistributed']:
        layer_group = h5
    else:
        layer_group = h5[layer_config['name']]
    layers = _get_h5_layers(layer_group)
    weights = layers['W'+BACKEND_SUFFIX]
    bias = layers['b'+BACKEND_SUFFIX]
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

def _time_distributed_parameters(h5, layer_config, n_in, layer_type):
    dist_layer = layer_config['layer']['config']
    dist_name = layer_config['layer']['class_name'].lower()
    subgroup = h5[layer_config['name']]
    return layer_converters[dist_name](subgroup, dist_layer, n_in, layer_type)

def _normalization_parameters(h5, layer_config, n_in, layer_type):
    """Get weights (gamma), bias (beta), for normalization layer"""
    layer_group = h5[layer_config['name']]
    layers = _get_h5_layers(layer_group)
    gamma = layers['gamma']
    beta = layers['beta']
    mean = layers['running_mean']
    stddev = layers['running_std']
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

def _get_maxout_layer_parameters(h5, layer_config, n_in, layer_type):
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

# TODO: unify LSTM, highway, and GRU here, they do almost the same thing
def _lstm_parameters(h5, layer_config, n_in, layer_type):
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
    return {'components': submap, 'architecture': 'lstm',
            'activation': _activation_map[layer_config['activation']],
            'inner_activation': _activation_map[layer_config['inner_activation']]}, n_out

def _get_highway_layer_parameters(h5, layer_config, n_in, layer_type):
    """Get weights, bias, and n-outputs for a highway layer"""
    layer_group = h5[layer_config['name']]
    layers = _get_h5_layers(layer_group)
    n_out = layers['W'].shape[1]

    submap = {}
    for gate in ['', '_carry']:
        submap[gate[1:] or 't'] = {
            'weights': layers['W' + gate].T.flatten().tolist(),
            'bias': layers['b' + gate].flatten().tolist(),
        }
    return {'components': submap, 'architecture': 'highway',
            'activation': _activation_map[layer_config['activation']]}, n_out

def _gru_parameters(h5, layer_config, n_in, layer_type):
    """GRU parameter converter"""
    layer_group = h5[layer_config['name']]
    layers = _get_h5_layers(layer_group)
    n_out = layers['W_h'].shape[1]

    submap = {}
    for gate in 'zrh':
        submap[gate] = {
            'U': layers['U_' + gate].T.flatten().tolist(),
            'weights': layers['W_' + gate].T.flatten().tolist(),
            'bias': layers['b_' + gate].flatten().tolist(),
        }
    return {'components': submap, 'architecture': 'gru',
            'activation': layer_config['activation'],
            'inner_activation': layer_config['inner_activation']}, n_out

def _get_merge_layer_parameters(h5, layer_config, n_in, layer_type):
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


def _get_elu_activation_parameters(h5, layer_config, n_in):
    """Return dummy parameters for ELU activation. Assert the alpha parameter is 1.0"""
    assert np.fabs(layer_config["alpha"]-1.0) < 1e-6
    return {'weights':[], 'bias':[], 'architecture':'dense','alpha':layer_config["alpha"],
            'activation':_activation_map['elu']}, n_in

def _activation_parameters(h5, layer_config, n_in, layer_type):
    """Return dummy parameters"""
    return {'weights':[], 'bias':[], 'architecture':'dense',
            'activation':_activation_map[layer_config['activation']]}, n_in

# _________________________________________________________________________
# master list of converters

layer_converters = {
    'dense': _get_dense_layer_parameters,
    'batchnormalization': _normalization_parameters,
    'highway': _get_highway_layer_parameters,
    'maxoutdense': _get_maxout_layer_parameters,
    'lstm': _lstm_parameters,
    'gru': _gru_parameters,
    'merge': _get_merge_layer_parameters,
    'activation': _activation_parameters,
    'timedistributed': _time_distributed_parameters,
    'elu': _get_elu_activation_parameters,
    }

# __________________________________________________________________________
# utilities

# utility function to handle keras layer naming
def _get_h5_layers(layer_group):
    """
    Keras: v1:
    For some reason Keras prefixes the datasets we need with the group
    name. This function returns a dictionary of the datasets, keyed
    with the group name stripped off.
    """
    group_name = layer_group.name.lstrip('/')
    strip_length = len(group_name) + 1
    prefixes = set()
    layers = {}
    for long_name, ds in layer_group.items():
        if long_name.startswith(group_name):
            name = long_name[strip_length:]
            prefixes.add(long_name[:strip_length])
        else:
            name_parts = long_name.split('_')
            assert name_parts[1].isnumeric()
            assert len(name_parts) > 2
            name = '_'.join(name_parts[2:])
            prefixes.add('_'.join(name_parts[:2]))
        layers[name] = np.asarray(ds)
    assert len(prefixes) == 1, prefixes
    return layers
