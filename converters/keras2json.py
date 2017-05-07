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
from keras_v1_layer_converters import layer_converters,\
 _send_recieve_meta_info
from keras_layer_converters_common import skip_layers

def _run():
    """Top level routine"""
    args = _get_args()
    with open(args.arch_file, 'r') as arch_file:
        arch = json.load(arch_file)
    with open(args.variables_file, 'r') as inputs_file:
        inputs = json.load(inputs_file)

    _check_version(arch)
    if arch["class_name"] != "Sequential":
        sys.exit("this is not a Sequential model, try using kerasfunc2json")

    with h5py.File(args.hdf5_file, 'r') as h5:
        out_dict = {
            'layers': _get_layers(arch, inputs, h5),
        }
        out_dict.update(_parse_inputs(inputs))
    print(json.dumps(out_dict, indent=2, sort_keys=True))

def _check_version(arch):
    global BACKEND
    if 'backend' not in arch:
        sys.stderr.write(
            'WARNING: no backend found for this architecture!\n'
            'Defaulting to theano.\n')
        BACKEND="theano"
    else:
        BACKEND = arch['backend']
    global KERAS_VERSION
    if 'keras_version' not in arch:
        sys.stderr.write(
            'WARNING: no version number found for this architecture!\n'
            'Defaulting to version 1.2.\n')
        KERAS_VERSION=1
    else:
        major, minor, *bugfix = arch['keras_version'].split('.')
        if major != '1' or minor < '2':
            warn_tmp = (
                "WARNNING: This converter was developed for Keras version 1.2. "
                "Your version (v{}.{}) may be incompatible.\n")
            sys.stderr.write(warn_tmp.format(major, minor))
        KERAS_VERSION=int(major)
    _send_recieve_meta_info(BACKEND)


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

def _get_layers(network, inputs, h5):
    layers = []
    in_layers = network['config']
    n_out = len(inputs['inputs'])
    for layer_n in range(len(in_layers)):
        # get converter for this layer
        layer_arch = in_layers[layer_n]
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
