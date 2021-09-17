#!/usr/bin/env python3
"""Utility to aid in regression tests

Reads from a pipe. The pipe data should be formatted as

  variable_name value1
  other_variable value2
  ...

with one variable per line.

Use as follows:
 - With no arguments, reformat the input as json and pipe to stdout
 - If a json file given as an argument, test for equality with stdin

"""

import argparse
import json
import sys
from collections.abc import Sequence

def _get_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('saved_variables', nargs='?')
    parser.add_argument('-t', '--tolerance', type=float, default=0.00001)
    parser.add_argument('-g', '--graph', action='store_false',
                        help="expect a graph regression test")
    return parser


def _run():
    parser = _get_args()
    args = parser.parse_args()
    if sys.stdin.isatty():
        parser.print_usage()
        sys.exit('you need to pipe in a file')

    parse = _get_output_node_dicts
    in_dict = parse(sys.stdin)
    if not args.saved_variables:
        print(json.dumps(in_dict, indent=2, sort_keys=True))
        sys.exit(1)

    if not in_dict:
        sys.stderr.write('Failure: Inputs were empty!\n')
        sys.exit(1)

    with open(args.saved_variables) as old_file:
        old_dict = json.load(old_file)

    # for the sequential tests we make a dummy output node name
    if not args.graph:
        in_dict = {'dummy': in_dict}
        old_dict = {'dummy': old_dict}

    good_nodes = set()
    for node_name, old_node in old_dict.items():
        input_node = in_dict[node_name]
        if _compare_equal(old_node, input_node, args.tolerance):
            good_nodes.add(node_name)
        else:
            sys.stderr.write('Failure!\n')
            sys.exit(1)

    sys.stdout.write('all outputs within thresholds!\n')
    sys.exit(0)

def _get_output_node_dicts(infile):
    """parse the input to get a nested dict, one for each output node

    We expect each node name to end with a `:` and include one
    entry. In other words, formatted as `<key>:`

    The values for each element in the node are formatted as
    `<key> <value>`.
    """
    odict = {}
    node_key = None
    for line in infile:
        if line.endswith(':\n'):
            node_key = line[:-1]
            assert node_key not in odict
            odict[node_key] = {}
            continue

        key, *vals = line.split()
        vals = [float(val) for val in vals]
        if node_key is not None:
            odict[node_key][key] = vals if len(vals) > 1 else vals[0]
        else:
            odict[key] = vals if len(vals) > 1 else vals[0]
    return odict


def _get_dict(infile):
    """Simpler version of the input stream parser. This expects several
    lines, formatted as `<key> <value>`.
    """
    odict = {}
    first_key = None
    for line in infile:
        try:
            key, val = line.split()
            odict[key] = [float(val)]
        except ValueError:
            # first line as key
            first_key = line.replace("\n", "")
        except Exception as e:
            print(e)
    if first_key is None:
      return odict
    else:
      return {first_key: odict}


def _compare_equal(old, new, tolerance, warn_threshold=0.0000001):
    """
    For values of x where abs(x) < 1, the threshold is absolute.
    For larger values, use a relative threshold.

    Assumes inputs are of same type and of dict, list or int/float

    """
    if isinstance(old, (int, float)):
        diff = old - new
        avg = (old + new) / 2
        rel = abs(diff) / abs(avg) if abs(avg) > 1 else abs(diff)
        # first do warnings
        if rel > warn_threshold:
            sys.stderr.write(
              'WARNING: value is off in new version by {}\n'.format(
               diff))
        if rel > tolerance:
            sys.stderr.write(
             'ERROR: change in value is over threshold {}\n'.format(
              tolerance))
            return False
        return True
    elif isinstance(old, list):
        correct = []
        for o, n in zip(old, new):
            correct.append(_compare_equal(o, n, tolerance))
        if False in correct:
            return False
        else:
            return True
    elif isinstance(old, dict):
        if set(old) != set(new):
            sys.stderr.write(
                'ERROR: variable mismatch. Targets: "{}", given "{}"\n'.format(
                    ', '.join(old), ', '.join(new)))
            return False
        correct = []
        for o_v, n_v in zip(old.values(), new.values()):
            correct.append(_compare_equal(o_v, n_v, tolerance))
        if False in correct:
            return False
        else:
            return True


if __name__ == '__main__':
    _run()
