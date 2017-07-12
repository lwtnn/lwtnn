# __________________________________________________________________________
# Common layer converter tools/methods
#

# _________________________________________________________________________
# master list of layers we skip
skip_layers = {'flatten', 'dropout', 'masking'}

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
    'hard_sigmoid': 'hard_sigmoid',
    'elu':'elu',
}
