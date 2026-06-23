# __________________________________________________________________________
# Common ONNX converter tools/methods
#

# _________________________________________________________________________
# ops we drop from the graph (shape-only / inference no-ops)
skip_ops = {
    'Identity',
    'Cast',
    'Dropout',
    'Flatten',
    'Reshape',
    'Squeeze',
    'Unsqueeze',
}

# translate from ONNX op name to lwtnn JSON activation
activation_map = {
    'Relu': 'rectified',
    'Sigmoid': 'sigmoid',
    'Tanh': 'tanh',
    'Softmax': 'softmax',
    'HardSigmoid': 'hard_sigmoid',
    # LeakyRelu / Elu carry an alpha and are handled by the op converter
}

# opset 13 promoted Squeeze/Unsqueeze axes from attribute to input;
# supporting both forms isn't worth the complexity
MIN_OPSET = 13
