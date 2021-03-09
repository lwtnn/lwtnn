#!/usr/bin/env python3
"""
Convert a keras model, saved with model.save(...) to a weights and
architecture component.
"""
import argparse

def get_args():
    d = '(default: %(default)s)'
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model')
    parser.add_argument('-w','--weight-file-name', default='weights.h5',
                        help=d)
    parser.add_argument('-a', '--architecture-file-name',
                        default='architecture.json', help=d)
    return parser.parse_args()

def run():
    args = get_args()
    from h5py import File
    import json

    m = File(args.model,'r')
    with File(args.weight_file_name,'w') as w:

        # the model weights should be moved to the root level of
        # output weights file
        weights = m['model_weights']
        for name, wt in weights.items():
            w.copy(wt, name)

        # we also need to add some information to the architecture
        # file, which is normally stored as model_weights attributes
        meta_keys = {'backend','keras_version'}
        try:
            meta = {x: str(weights.attrs[x],'UTF-8') for x in meta_keys}
        except TypeError:
            meta = {x: weights.attrs[x] for x in meta_keys}

        # also seems nessisary to store some attributes on the weights
        # file
        for a in ['layer_names']:
            w.attrs[a] = weights.attrs[a]

    arch = json.loads(m.attrs['model_config'])
    arch.update(meta)
    with open(args.architecture_file_name,'w') as arch_file:
        arch_file.write(json.dumps(arch,indent=2))


if __name__ == '__main__':
    run()
