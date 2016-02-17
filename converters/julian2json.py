#!/usr/bin/env python3
"""Converter from Julian's npy files to JSON file for Athena

At the moment we assume that numpy parameter files are named
`[bw]_l_*.npy`, where `b` indicates a bias layer, `w` a weights layer,
and `*` is the layer number. We also assume the `arch_dir` contains
`arch.json`, `mean.npy`, and `std.npy`.
"""

# doc strings for options
_parameters_dir_help="directory containing numpy parameter files"
_arch_dir_help="directory containing preprocessing and arch files"
_nopre_help="don't do any preprocessing (arch_dir can be anything)"
_summarize_help=("summarize the weights and bias dimensions. "
                 "Won't produce a usable input for lwtnn.")

# hardcoded file names
_arch_template='{}/arch.json'
_mean_template='{}/mean.npy'
_std_template='{}/std.npy'

import argparse, glob, os, re
import json
import numpy as np

def _run():
    """Top level routine"""
    args = _get_args()

    layers = _get_layers(args.parameters_dir, args.arch_dir)
    json_layers = _layers_to_json(layers, args.summarize)

    if args.no_pre:
        inputs, defaults = _get_inputs_no_preproc(args.arch_dir)
    else:
        inputs, defaults = _get_inputs(args.arch_dir)
    outputs = _get_outputs(layers, args.arch_dir)

    out_dict = {
        'layers': json_layers,
        'inputs': inputs,
        'outputs': outputs,
        'defaults': defaults
        }
    print(json.dumps(out_dict, indent=2))

def _get_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('parameters_dir', help=_parameters_dir_help)
    parser.add_argument('arch_dir', help=_arch_dir_help)
    parser.add_argument('-s', '--summarize', action='store_true',
                        help=_summarize_help)
    parser.add_argument('--no-pre', action='store_true', help=_nopre_help)
    return parser.parse_args()

# __________________________________________________________________________
# build layers
def _get_layers(parameters_dir, arch_dir):
    with open(_arch_template.format(arch_dir)) as jfile:
        layer_meta = json.load(jfile)['layers']

    globstr = "{}/w_l_*.npy".format(parameters_dir)
    num_match = re.compile('[wb]_l_([0-9]+)\.npy')
    matches = glob.glob(globstr)
    assert len(layer_meta) == len(matches), 'number of layers mismatch'
    layers = [None]*len(matches)
    for weight_name in matches:
        dirname, basename = os.path.split(weight_name)
        number = int(num_match.search(basename).group(1))
        weight = np.load(weight_name)
        bias = np.load(os.path.join(dirname, 'b' + basename[1:]))
        activation = layer_meta[number]['activation']
        layers[number] = (weight, bias, activation)
    return layers

def _layers_to_json(in_layers, summarize=False):
    layers = []
    last_layer = len(in_layers) - 1
    n_out = in_layers[0][0].shape[0]
    for number, (wt, bias, activation) in enumerate(in_layers):
        # sanity checks: this format expects y = x M + b style layers
        # so the number of rows in the current matrix must match the
        # number of columns in the previous matrix
        assert wt.shape[1] == bias.shape[0]
        assert wt.shape[0] == n_out
        n_out = wt.shape[1]

        if summarize:
            out_layer = {
                'activation': activation,
                'weights': list(wt.shape),
                'bias': list(bias.shape),
            }
        else:
            out_layer = {
                'activation': activation,
                # note that this format expects y = x M + b layers,
                # whereas lwtnn does y = M x + b layers. We must
                # transpose the weights to account for this
                'weights': wt.T.flatten('C').tolist(),
                'bias': bias.flatten('C').tolist(),
            }
        layers.append(out_layer)
    return layers

# __________________________________________________________________________
# build inputs
def _get_inputs_no_preproc(arch_dir):
    with open(_arch_template.format(arch_dir)) as jfile:
        inputs = json.load(jfile)['inputs']
    py_inputs = []
    for name in inputs:
        var_dic = {'name': name, 'offset': 0, 'scale': 1}
        py_inputs.append(var_dic)
    return py_inputs, {}

def _get_inputs(arch_dir):
    """Get the input names, offsets, and default values"""

    # read in the input names from the arch file
    with open(_arch_template.format(arch_dir)) as jfile:
        inputs = json.load(jfile)['inputs']

    # read in julian's inputs
    means = np.load(_mean_template.format(arch_dir))
    stdev = np.load(_std_template.format(arch_dir))
    assert len(inputs) == len(stdev) == len(means)
    py_inputs = []
    defaults = {}
    for number, name in enumerate(inputs):
        var_dic = {
            'name': name,
            'offset': -float(means[number]),
            'scale': 1 / float(stdev[number]),
            }
        defaults[name] = float(means[number])
        py_inputs.append( var_dic )
    return py_inputs, defaults

def _get_outputs(layers, arch_dir):
    """Return the output labels for Julian's nns"""

    with open(_arch_template.format(arch_dir)) as jfile:
        outputs = json.load(jfile)['outputs']
    assert len(outputs) == layers[-1][0].shape[1], 'n-outputs mismatch'
    return outputs

if __name__ == '__main__':
    _run()
