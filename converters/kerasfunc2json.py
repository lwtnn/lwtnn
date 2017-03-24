#!/usr/bin/env python3
#
# Converter from Keras sequential NN to JSON
"""____________________________________________________________________
Variable specification file

In additon to the standard Keras architecture and weights files, you
must provide a "variable specification" json file with the following
format:

  {
    "inputs": [
      {"name": variable_name,
       "scale": scale,
       "offset": offset,
       "default": default_value},
      ...
      ],
    "class_labels": [output_class_1_name, output_class_2_name, ...],
    "miscellaneous": {"key": "value"}
  }

where `scale` and `offset` account for any scaling and shifting to the
input variables in preprocessing. The "default" value is optional.

The "miscellaneous" object is also optional and can contain (key,
value) pairs of strings to pass to the application.

"""

import argparse
import json
import h5py
from collections import Counter
import sys
from keras_layer_converters import layer_converters, skip_layers

def _run():
    """Top level routine"""
    args = _get_args()
    with open(args.arch_file, 'r') as arch_file:
        arch = json.load(arch_file)
    with open(args.variables_file, 'r') as inputs_file:
        inputs = json.load(inputs_file)

    _check_version(arch)
    if arch["class_name"] != "Model":
        sys.exit("this is not a graph, try using keras2json")

    with h5py.File(args.hdf5_file, 'r') as h5:
        out_dict = {
            'layers': _get_layers(arch, h5),
        }
        out_dict.update(_parse_inputs(inputs))
    print(json.dumps(out_dict, indent=2, sort_keys=True))

def _check_version(arch):
    if 'keras_version' not in arch:
        sys.stderr.write(
            'WARNING: no version number found for this archetecture!\n')
        return
    major, minor, *bugfix = arch['keras_version'].split('.')
    if major != '1' or minor < '2':
        warn_tmp = (
            "WARNNING: This converter was developed for Keras version 1.2. "
            "Your version (v{}.{}) may be incompatible.\n")
        sys.stderr.write(warn_tmp.format(major, minor))

def _get_args():
    parser = argparse.ArgumentParser(
        description="Converter from Keras saved NN to JSON",
        epilog=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('arch_file', help='architecture json file')
    parser.add_argument('variables_file', help='variable spec as json')
    parser.add_argument('hdf5_file', help='Keras weights file')
    return parser.parse_args()


# __________________________________________________________________________
# master layer converter / inputs function

# output names
LAYERS = 'layers'
NODES = 'nodes'

_merge_layers = set(['concat'])

class Node:
    def __init__(self, layer, idx):
        self.layer_type = layer['class_name'].lower()
        self.name = layer['name']
        self.idx = idx
        self.sources = []
        self.number = None
        self.n_outputs = None
        inbound = layer['inbound_nodes']
        if self.layer_type != "inputlayer":
            for sname, sidx, something in inbound[idx]:
                assert something == 0
                self.sources.append( (sname, sidx) )
        else:
            self.n_outputs = layer['config']['batch_input_shape'][1]
    def __str__(self):
        parents = ', '.join(( str(x) for x in self.sources))
        tmp = '{} <- [{}]'
        if self.number is not None:
            tmp = '{}{{{n}}} <- [{}]'
        return tmp.format( (self.name, self.idx), parents, n=self.number)
    def get_key(self):
        return (self.name, self.idx)
    def __lt__(self, other):
        return self.get_key() < other.get_key()

# We want to build one node per _inbound_ node
def _build_node_dict(network):
    layers = {l['name']: l for l in network['config']['layers']}
    nodes = {}
    for top_name, top_layer in layers.items():
        if (top_name, 0) not in nodes:
            nodes[(top_name, 0)] = Node(top_layer, 0)
        for sink in top_layer['inbound_nodes']:
            for merge_node in sink:
                lname, idx, something = merge_node
                assert something == 0
                layer = layers[lname]
                id_tup = (lname, idx)
                if id_tup not in nodes:
                    nodes[id_tup] = Node(layer, idx)

    # now we collapse the node references
    for node in nodes.values():
        source_nodes = []
        for source in node.sources:
            source_nodes.append(nodes[source])
        node.sources = source_nodes
    return nodes


def _number_nodes(node_dict):
    for number, node in enumerate(sorted(node_dict.values())):
        node.number = number

def _build_layer(layer_dict, node_name, h5, node_dict):
    node = node_dict[node_name]
    if node.n_outputs is not None:
        return

    for source in node.sources:
        _build_layer(layer_dict, source.get_key(), h5, node_dict)

    # special cases for merge layers
    if node.layer_type == 'merge':
        node.n_outputs = 0
        for source in node.sources:
            node.n_outputs += source.n_outputs
        return
    

def _get_layers(network, h5):
    node_dict = _build_node_dict(network)
    _number_nodes(node_dict)
    for node in node_dict.values():
        print(str(node))

    layer_dict = {}
    for node_name in set(node_dict):
        _build_layer(layer_dict, node_name, h5, node_dict)
        # get converter for this layer
        # TODO: move this into the build_layer function
        layer_type = layer_arch['class_name'].lower()
        if layer_type in skip_layers: continue
        convert = layer_converters[layer_type]

        # build the out layer
        out_layer, n_out = convert(h5, layer_arch['config'], n_out)
        layers.append(out_layer)
    return layers

def _parse_inputs(keras_dict):
    inputs = []
    defaults = {}
    for val in keras_dict['inputs']:
        inputs.append({x: val[x] for x in ['offset', 'scale', 'name']})

        # maybe fill default
        default = val.get("default")
        if default is not None:
            defaults[val['name']] = default
    out = {
        'inputs': inputs,
        'outputs': keras_dict['class_labels'],
        'defaults': defaults,
    }
    if 'miscellaneous' in keras_dict:
        misc_dict = {}
        for key, val in keras_dict['miscellaneous'].items():
            misc_dict[str(key)] = str(val)
        out['miscellaneous'] = misc_dict
    return out

if __name__ == '__main__':
    _run()
