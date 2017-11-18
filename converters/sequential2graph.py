#!/usr/bin/env python3

"""
Convert a sequential model to a graph model
"""
_input_help ='read from stdin if no file is given'

import argparse
import json
import sys

def get_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input_file', nargs='?', help=_input_help)
    return parser.parse_args()

def run():
    args = get_args()
    if not args.input_file:
        if sys.stdin.isatty():
            exit('need an input file')
        input_file = sys.stdin
    else:
        input_file = open(args.input_file)

    input_nn = json.load(input_file)
    n_layers = len(input_nn['layers'])
    output_nn = {
        "inputs": funcify_inputs(input_nn['inputs'], input_nn['defaults']),
        "input_sequences": [],
        "layers": input_nn['layers'],
        "nodes": build_nodes(n_layers, len(input_nn['inputs'])),
        "outputs": {
            "output_node": {
                "labels": input_nn['outputs'],
                "node_index": n_layers
            }
        }
    }
    json.dump(output_nn, sys.stdout, indent=2)

def funcify_inputs(inputs, defaults):
    new_inputs = []
    for input_var in inputs:
        new_input = input_var.copy()
        input_name = new_input['name']
        if input_name in defaults:
            new_input['default'] = defaults[input_name]
        new_inputs.append(new_input)
    return [{"name": "input_node", "variables": new_inputs}]

def build_nodes(n_layers, n_inputs):
    layers = [
        {
            "type": "input",
            "size": n_inputs,
            "sources": [ 0 ],
        }
    ]
    for lay_n in range(n_layers):
        new_layer = {
            "type": "feed_forward",
            "layer_index": lay_n,
            "sources": [lay_n],
        }
        layers.append(new_layer)
    return layers

if __name__ == '__main__':
    run()
