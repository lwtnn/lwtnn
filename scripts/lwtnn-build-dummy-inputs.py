#!/usr/bin/env python3

"""Generate fake serialized NNs to test the lightweight classes"""

import argparse
import json
import h5py
import numpy as np

def _run():
    args = _get_args()
    _build_keras_arch("arch.json")
    _build_keras_inputs_file("variable_spec.json")
    _build_keras_weights("weights.h5", verbose=args.verbose)

def _get_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-v', '--verbose', action='store_true')
    return parser.parse_args()

def _build_keras_arch(name):
    arch = {
        'layers': [
            {'activation': 'relu', 'name': 'Dense'}
        ]
    }
    with open(name, 'w') as out_file:
        out_file.write(json.dumps(arch, indent=2))

def _build_keras_inputs_file(name):
    def build_input(num):
        return {"name": "in{}".format(num), "offset": 0.0, "scale": 1.0}
    top = {
        "inputs": [build_input(x) for x in range(1,5)],
        "class_labels": ["out{}".format(x) for x in range(1,5)]
    }
    with open(name, 'w') as out_file:
        out_file.write(json.dumps(top, indent=2))

def _build_keras_weights(name, verbose):
    half_swap = np.zeros((4,4))
    half_swap[0,3] = 1.0
    half_swap[1,2] = 1.0
    if verbose:
        print(half_swap)

    bias = np.zeros(4)
    with h5py.File(name, 'w') as h5_file:
        layer0 = h5_file.create_group("layer_0")
        layer0.create_dataset("param_0", data=half_swap)
        layer0.create_dataset("param_1", data=bias)

if __name__ == "__main__":
    _run()
