#!/usr/bin/env python3
"""
Converter from AiglePack Yaml file to JSON file for Athena
"""

import yaml
import argparse
import json

def _run():
    """Top level routine"""
    args = _get_args()
    with open(args.yaml_file) as yml:
        network = yaml.load(yml)['network']
    out_dict = {
        'layers': _get_layers(network),
        'inputs': _get_inputs(network),
        'outputs': _get_outputs(network)
        }
    defaults = network.get('defaults')
    if defaults:
        out_dict['defaults'] = defaults
    print(json.dumps(out_dict))

def _get_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('yaml_file')
    return parser.parse_args()

def _get_layers(network):
    layers = []
    for layer_name in network['layer_access']:
        layer = network[layer_name]
        out_layer = {
            'activation': layer['activation'],
            'weights': _line_to_array(layer['weights']),
            'bias': _line_to_array(layer['bias']),
            'architecture': 'dense'
        }
        layers.append(out_layer)
    return layers

def _line_to_array(agile_line):
    """convert the weird AGILEPack weights output to an array of floats"""
    entries = agile_line.split(',')[1:]
    return [float(x) for x in entries]

def _get_inputs(network):
    """
    Get the input scaling from AGILEPack.
    Note the inversion of scale and offset.
    """
    inputs = []
    for input_name in network['input_order']:
        offset = - network['scaling']['mean'][input_name]
        scale = 1 / network['scaling']['sd'][input_name]
        inputs.append({'name': input_name, 'offset': offset, 'scale': scale})
    return inputs

def _get_outputs(network):
    return network['target_order']

if __name__ == '__main__':
    _run()
