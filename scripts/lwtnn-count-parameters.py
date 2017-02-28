#!/usr/bin/env python3

"""
Utility to count the number of free parameters in a saved network
"""
_help_subset='only consider a subset of the configuration'


import json, sys
from argparse import ArgumentParser
from collections import Mapping, Sequence, Counter
from numbers import Number, Integral

def count_numbers(node, header='number', all_numbers=True):
    numbers = Counter()
    if isinstance(node, Mapping):
        for subheader, subnode in node.items():
            numbers += count_numbers(subnode, subheader, all_numbers)
    elif isinstance(node, Sequence) and len(node) > 0 and node[0] != node:
        for subnode in node:
            numbers += count_numbers(subnode, header, all_numbers)
    elif isinstance(node, Number):
        if all_numbers:
            numbers[header] += 1
        else:
            if not isinstance(node, Integral):
                numbers[header] += 1
    return numbers

def _get_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('nn_file')
    parser.add_argument(
        '-s', '--subset', const='layers', help=_help_subset, nargs='?',
        metavar='SUBSET=layers')
    parser.add_argument('-i', '--include-integers', action='store_true')
    return parser.parse_args()

def run():
    args = _get_args()
    with open(args.nn_file, 'r') as infile:
        nn = json.load(infile)

    if args.subset:
        nn = nn[args.subset]
    counts = count_numbers(nn, all_numbers=args.include_integers)
    sum_numbers = 0
    for heading, number in counts.items():
        print('{}: {}'.format(heading, number))
        sum_numbers += number
    print('TOTAL: {}'.format(sum_numbers))

if __name__ == '__main__':
    run()
