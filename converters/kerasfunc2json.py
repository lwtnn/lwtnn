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
    with open(args.variables_file, 'r') as variables_file:
        variables = json.load(variables_file)

    _check_version(arch)
    if arch["class_name"] != "Model":
        sys.exit("this is not a graph, try using keras2json")

    with h5py.File(args.hdf5_file, 'r') as h5:
        layers, node_dict = _get_layers_and_nodes(arch, h5)
    input_layer_arch = arch['config']['input_layers']
    nodes = _build_output_node_list(node_dict, input_layer_arch)

    out_dict = {
        'layers': layers, 'nodes': nodes,
        'inputs': _parse_inputs(
            variables['inputs'],
            _get_vars_per_input(input_layer_arch, node_dict)),
        'outputs': _parse_outputs(
            variables['outputs'],
            arch['config']['output_layers'],
            node_dict),
    }
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
    # TODO: make the node object a wrapper on the Keras layer dictionary
    def __init__(self, layer, idx):
        self.layer_type = layer['class_name'].lower()
        self.name = layer['name']
        self.idx = idx
        self.sources = []
        self.number = None
        self.layer_number = None
        self.n_outputs = None
        inbound = layer['inbound_nodes']
        if self.layer_type != "inputlayer":
            for sname, sidx, something in inbound[idx]:
                assert something == 0
                self.sources.append( (sname, sidx) )
        else:
            self.n_outputs = layer['config']['batch_input_shape'][1]
        self.keras_layer = layer

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

def _build_layer(output_layers, node_key, h5, node_dict, layer_dict):
    node = node_dict[node_key]
    if node.n_outputs is not None:
        return

    for source in node.sources:
        _build_layer(output_layers, source.get_key(), h5,
                     node_dict, layer_dict)

    # special cases for merge layers
    if node.layer_type == 'merge':
        node.n_outputs = 0
        for source in node.sources:
            node.n_outputs += source.n_outputs
        return

    assert len(node.sources) == 1

    # if this layer is already defined, just add the node count and
    # continue
    if node.name in layer_dict:
        node.n_outputs = layer_dict[node.name]['n_outputs']
        node.layer_number = layer_dict[node.name]['pos']
        return

    layer_type = node.layer_type
    if layer_type in skip_layers:
        return
    convert = layer_converters[layer_type]

    # build the out layer
    n_inputs = sum(s.n_outputs for s in node.sources)
    layer_config = node.keras_layer['config']
    out_layer, node.n_outputs = convert(h5, layer_config, n_inputs)
    layer_number = len(output_layers)
    layer_dict[node.name] = {
        'n_outputs': node.n_outputs,
        'pos': layer_number}
    node.layer_number = layer_number
    output_layers.append(out_layer)

_node_type_map = {
    'merge': 'concatenate',
    'inputlayer': 'input',
    'dense': 'feed_forward',
}

def _build_output_node_list(node_dict, input_layer_arch):
    """
    no effort is made to sort this list in any way, but the ordering
    is important because each node contains indices for other nodes
    """
    node_list = []
    input_map = {n[0]:i for i, n in enumerate(input_layer_arch)}
    for node in sorted(node_dict.values()):
        node_type = _node_type_map[node.layer_type]
        out_node = {'type': node_type}
        if node.sources:
            out_node['sources'] = [n.number for n in node.sources]

        if node_type == 'input':
            out_node['sources'] = [input_map[node.name]]
            out_node['size'] = node.n_outputs
        elif node_type == 'feed_forward':
            out_node['layer_index'] = node.layer_number
        node_list.append(out_node)
    return node_list

def _get_layers_and_nodes(network, h5):
    node_dict = _build_node_dict(network)
    _number_nodes(node_dict)

    output_layers = []
    layer_meta = {}
    for node_key in node_dict:
        _build_layer(output_layers, node_key, h5, node_dict, layer_meta)

    return output_layers, node_dict

def _get_vars_per_input(input_layer_arch, node_dict):
    vars_per_input = {}
    for nodenum, (lname, lidx, something) in enumerate(input_layer_arch):
        assert lidx == 0 and something == 0
        vars_per_input[nodenum] = node_dict[(lname, lidx)].n_outputs
    return vars_per_input

def _parse_inputs(input_list, vars_per_input):
    nodes = []
    for input_number, node in enumerate(input_list):
        inputs = []
        node_name = node['name']
        for val in node['variables']:
            var_info = {x: val[x] for x in ['offset', 'scale', 'name']}
            default = val.get("default")
            if default is not None:
                var_info['default'] = default
            inputs.append(var_info)

        assert vars_per_input[input_number] == len(inputs)

        nodes.append({'name': node_name, 'variables': inputs})
    return nodes

def _parse_outputs(user_outputs, output_layers, node_dict):
    outputs = {}
    assert len(user_outputs) == len(output_layers)
    for num, (usr, ker) in enumerate(zip(user_outputs, output_layers)):
        kname, kid, ks = ker
        assert kid == 0 and ks == 0
        node = node_dict[(kname, kid)]
        assert node.n_outputs == len(usr['labels'])
        assert usr['name'] not in outputs
        output = {
            'node_index': node.number,
            'labels': usr['labels']
        }
        outputs[usr['name']] = output
    return outputs

def _parse_miscellaneous(variables):
    if 'miscellaneous' in variables:
        misc_dict = {}
        for key, val in variables['miscellaneous'].items():
            misc_dict[str(key)] = str(val)
        out['miscellaneous'] = misc_dict
    return out

if __name__ == '__main__':
    _run()
