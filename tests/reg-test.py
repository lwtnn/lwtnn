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

def _run():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('saved_variables', nargs='?')
    parser.add_argument('-t', '--tolerance', type=float, default=0.0001)
    args = parser.parse_args()

    if sys.stdin.isatty():
        parser.print_usage()
        sys.exit('you need to pipe in a file')

    in_dict = _get_dict(sys.stdin)
    if not args.saved_variables:
        print(json.dumps(in_dict, indent=2, sort_keys=True))
        sys.exit(1)

    with open(args.saved_variables) as old_file:
        old_dict = json.load(old_file)
    if _compare_equal(old_dict, in_dict, args.tolerance):
        sys.stdout.write('all outputs within thresholds!\n')
        sys.exit(0)
    else:
        sys.stderr.write('Failure!\n')
        sys.exit(1)

def _get_dict(infile):
    odict = {}
    for line in infile:
        key, val = line.split()
        odict[key] = float(val)
    return odict

def _compare_equal(old, new, tolerance, warn_threshold=0.000001):
    """
    For values of x where abs(x) < 1, the threshold is absolute.
    For larger values, use a relative threshold.
    """
    if set(old) != set(new):
        sys.stderr.write(
            'ERROR: variable mismatch. Targets: "{}", given "{}"\n'.format(
                ', '.join(old), ', '.join(new)))
        return False

    fails = set()
    for var in old:
        diff = old[var] - new[var]
        avg = (old[var] + new[var]) / 2
        rel = abs(diff) / abs(avg) if abs(avg) > 1 else abs(diff)
        # first do warnings
        if rel > warn_threshold:
            sys.stderr.write(
                'WARNING: "{}" is off in new version by {}\n'.format(
                    var, diff))
        if rel > tolerance:
            sys.stderr.write(
                'ERROR: change in "{}" is over threshold {}\n'.format(
                    var, tolerance))
            fails.add(var)
    if fails:
        return False
    return True

if __name__ == '__main__':
    _run()
