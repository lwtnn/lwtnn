#!/usr/bin/env python3

"""
Build test patterns for lwtnn-test-arbitrary-net

Givan a variable specification file, will produce two files:
 - A file specifying the names of the inputs
 - A file containing dummy input variables

"""

from argparse import ArgumentParser, RawDescriptionHelpFormatter
import json

def _get_args():
    parser = ArgumentParser(
        description=__doc__,
        formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument('inputs_file')
    parser.add_argument('-n','--names-file', default='input_names.txt')
    parser.add_argument('-v','--vals-file', default='input_values.txt')
    return parser.parse_args()

def run():
    args = _get_args()
    with open(args.inputs_file) as inputs_file:
        inputs = json.loads(''.join(inputs_file.readlines()))

    input_names = [x['name'] for x in inputs['inputs']]
    with open(args.names_file, 'w') as inputs_file:
        inputs_file.write(' '.join(input_names))

    dummy_vals = [1 for _ in input_names]
    with open(args.vals_file, 'w') as vals_file:
        vals_file.write(' '.join([str(x) for x in dummy_vals]))

if __name__ == '__main__':
    run()
