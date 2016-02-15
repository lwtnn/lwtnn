#!/usr/bin/env python3
"""Converter from Julian's npy files to JSON file for Athena

At the moment we assume that numpy files are named `{b,w}_l_X.npy`.
"""

import argparse, glob
import json
import numpy as np
import os

def _run():
    """Top level routine"""
    args = _get_args()

    layers = _get_layers(args.parameters_dir)
    json_layers = _layers_to_json(layers)

    inputs, defaults = _get_inputs(args.preproc_dir)
    outputs = _get_outputs(layers)

    out_dict = {
        'layers': json_layers,
        'inputs': inputs,
        'outputs': outputs,
        'defaults': defaults
        }
    print(json.dumps(out_dict, indent=2))

def _get_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('parameters_dir')
    parser.add_argument('preproc_dir')
    return parser.parse_args()

# __________________________________________________________________________
# build layers
def _get_layers(parameters_dir):
    globstr = "{}/w_l_*.npy".format(parameters_dir)
    layers = []
    for weight_name in glob.glob(globstr):
        dirname, basename = os.path.split(weight_name)
        number = int(basename[-5])
        weight = np.load(weight_name)
        bias = np.load(os.path.join(dirname, 'b' + basename[1:]))
        layers.append( (weight, bias) )
    return layers

def _layers_to_json(in_layers):
    layers = []
    last_layer = len(in_layers) - 1
    n_out = in_layers[0][0].shape[0]
    for number, (wt, bias) in enumerate(in_layers):
        assert wt.shape[1] == bias.shape[0]
        assert wt.shape[0] == n_out
        n_out = wt.shape[1]
        out_layer = {
            'activation': 'linear' if number != last_layer else 'sigmoid',
            'weights': wt.flatten('C').tolist(),
            'bias': bias.flatten('C').tolist(),
        }
        layers.append(out_layer)
    return layers

# __________________________________________________________________________
# build inputs
def _get_inputs(inputs_dir):
    """
    The inputs from julian are (currently) hardcoded
    """
    inputs = ['pt', 'eta'] + [
        'track_2_d0_significance',
        'track_3_d0_significance',
        'track_2_z0_significance',
        'track_3_z0_significance',
        'n_tracks_over_d0_threshold',
        'jet_prob',
        'jet_width_eta',
        'jet_width_phi'] + [
            'vertex_significance', 'n_secondary_vertices',
            'n_secondary_vertex_tracks', 'delta_r_vertex',
            'vertex_mass', 'vertex_energy_fraction']

    # read in julian's inputs
    means = np.load("{}/high_mean.npy".format(inputs_dir))
    stdev1 = np.load("{}/high_std1.npy".format(inputs_dir))
    stdev2 = np.load("{}/high_std2.npy".format(inputs_dir))
    stdev = stdev1 * stdev2
    assert len(inputs) == len(stdev) == len(means)
    layers = []
    defaults = {}
    for number, name in enumerate(inputs):
        var_dic = {
            'name': name,
            'offset': -means[number],
            'scale': 1 / stdev[number],
            }
        defaults[name] = means[number]
        layers.append( var_dic )
    return layers, defaults

def _get_outputs(layers):
    n_outputs = layers[-1][0].shape[1]
    return ['out_{}'.format(i) for i in range(n_outputs)]

if __name__ == '__main__':
    _run()
