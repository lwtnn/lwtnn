# __________________________________________________________________________
# Op converters
#
# Each takes:
#  - initializers: dict {tensor_name: numpy.ndarray}
#  - onnx_node:    the onnx.NodeProto being converted
#  - n_in:         number of incoming features (for error checking)
#  - extra:        op-specific extras (e.g. the Add fused into MatMul)
#
# Each returns:
#  - a dict of layer parameters, serialisable to JSON
#  - the number of outputs (also for error checking)

import sys

import numpy as np

from onnx_converter_common import activation_map


def _attr(node, name, default=None):
    for a in node.attribute:
        if a.name != name:
            continue
        if a.type == 1:
            return a.f
        if a.type == 2:
            return a.i
        if a.type == 3:
            return a.s.decode()
        if a.type == 6:
            return list(a.floats)
        if a.type == 7:
            return list(a.ints)
    return default


def _gemm_parameters(initializers, node, n_in, extra=None):
    b_name = node.input[1]
    c_name = node.input[2] if len(node.input) > 2 else None

    if b_name not in initializers:
        sys.exit("Gemm '{}' has a non-constant weight matrix ({})".format(
            node.name, b_name))
    weights = initializers[b_name]
    bias = (initializers[c_name] if c_name is not None
            else np.zeros(weights.shape[-1], dtype=weights.dtype))

    trans_b = _attr(node, 'transB', 0)
    alpha = _attr(node, 'alpha', 1.0)
    beta = _attr(node, 'beta', 1.0)

    # lwtnn stores weights as (n_out, n_in) and dots rows against the input
    w = weights.T if not trans_b else weights
    if alpha != 1.0:
        w = w * alpha
    if beta != 1.0:
        bias = bias * beta

    assert w.shape[1] == n_in, (
        "Gemm '{}' expects {} inputs, got {}".format(
            node.name, w.shape[1], n_in))
    assert w.shape[0] == bias.shape[0]

    return {
        'weights': w.flatten('C').tolist(),
        'bias': bias.flatten('C').tolist(),
        'architecture': 'dense',
        'activation': 'linear',
    }, w.shape[0]


def _matmul_add_parameters(initializers, node, n_in, extra=None):
    # `node` is the MatMul, `extra` is the constant-bias Add that consumes it.
    # torch.onnx.export emits this pattern instead of Gemm for nn.Linear in
    # some configurations.
    b_name = node.input[1]
    if b_name not in initializers:
        sys.exit("MatMul '{}' has a non-constant weight matrix ({})".format(
            node.name, b_name))
    weights = initializers[b_name]

    bias_name = next((i for i in extra.input if i in initializers), None)
    if bias_name is None:
        sys.exit("Add '{}' fused with MatMul '{}' has no constant bias".format(
            extra.name, node.name))
    bias = initializers[bias_name]

    w = weights.T
    assert w.shape[1] == n_in, (
        "MatMul '{}' expects {} inputs, got {}".format(
            node.name, w.shape[1], n_in))
    assert w.shape[0] == bias.shape[-1]

    return {
        'weights': w.flatten('C').tolist(),
        'bias': bias.flatten('C').tolist(),
        'architecture': 'dense',
        'activation': 'linear',
    }, w.shape[0]


def _batchnorm_parameters(initializers, node, n_in, extra=None):
    # ONNX inputs are (X, scale, B, mean, var); we collapse to a single
    # affine that lwtnn's normalization layer applies.
    scale = initializers[node.input[1]]
    beta = initializers[node.input[2]]
    mean = initializers[node.input[3]]
    var = initializers[node.input[4]]
    epsilon = _attr(node, 'epsilon', 1e-5)

    assert scale.shape == beta.shape == mean.shape == var.shape
    assert scale.shape[0] == n_in

    weight = scale / np.sqrt(var + epsilon)
    bias = beta - mean * weight

    return {
        'weights': weight.flatten('C').tolist(),
        'bias': bias.flatten('C').tolist(),
        'architecture': 'normalization',
    }, weight.shape[0]


def _activation_layer(activation_name):
    def func(initializers, node, n_in, extra=None):
        return {
            'weights': [], 'bias': [], 'architecture': 'dense',
            'activation': activation_name,
        }, n_in
    return func


def _leaky_relu_parameters(initializers, node, n_in, extra=None):
    alpha = _attr(node, 'alpha', 0.01)
    return {
        'weights': [], 'bias': [], 'architecture': 'dense',
        'activation': {'function': 'leakyrelu', 'alpha': alpha},
    }, n_in


def _elu_parameters(initializers, node, n_in, extra=None):
    alpha = _attr(node, 'alpha', 1.0)
    return {
        'weights': [], 'bias': [], 'architecture': 'dense',
        'activation': {'function': 'elu', 'alpha': alpha},
    }, n_in


# _________________________________________________________________________
# master list of op converters

op_converters = {
    'Gemm': _gemm_parameters,
    'MatMul': _matmul_add_parameters,
    'BatchNormalization': _batchnorm_parameters,
    'Relu': _activation_layer(activation_map['Relu']),
    'Sigmoid': _activation_layer(activation_map['Sigmoid']),
    'Tanh': _activation_layer(activation_map['Tanh']),
    'Softmax': _activation_layer(activation_map['Softmax']),
    'HardSigmoid': _activation_layer(activation_map['HardSigmoid']),
    'LeakyRelu': _leaky_relu_parameters,
    'Elu': _elu_parameters,
}
