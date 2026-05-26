#!/usr/bin/env python3
# Fixture generator for test-onnx-dense.sh.
# Writes <out>/dense.onnx and <out>/variables.json.
import json
import os
import sys

import numpy as np
from onnx import TensorProto, helper, numpy_helper, save_model


LAYER_SIZES = [3, 5, 4, 2]
ACTIVATIONS = ['Relu', 'Relu', 'Softmax']
SEED = 42


def _build_model():
    rng = np.random.default_rng(SEED)
    nodes = []
    initializers = []
    cur = 'input'
    for i, (n_in, n_out) in enumerate(zip(LAYER_SIZES, LAYER_SIZES[1:])):
        w = rng.standard_normal((n_in, n_out)).astype(np.float32)
        b = rng.standard_normal((n_out,)).astype(np.float32)
        w_name, b_name = 'W{}'.format(i), 'B{}'.format(i)
        initializers += [
            numpy_helper.from_array(w, w_name),
            numpy_helper.from_array(b, b_name),
        ]
        gemm_out = 'gemm{}'.format(i)
        nodes.append(helper.make_node(
            'Gemm', [cur, w_name, b_name], [gemm_out],
            name='Gemm_{}'.format(i)))
        act = ACTIVATIONS[i]
        is_last = (i == len(LAYER_SIZES) - 2)
        act_out = 'output' if is_last else 'act{}'.format(i)
        attrs = {}
        if act == 'Softmax':
            attrs['axis'] = -1
        nodes.append(helper.make_node(
            act, [gemm_out], [act_out], name='{}_{}'.format(act, i), **attrs))
        cur = act_out

    graph = helper.make_graph(
        nodes, 'dense_test',
        inputs=[helper.make_tensor_value_info(
            'input', TensorProto.FLOAT, ['batch', LAYER_SIZES[0]])],
        outputs=[helper.make_tensor_value_info(
            'output', TensorProto.FLOAT, ['batch', LAYER_SIZES[-1]])],
        initializer=initializers,
    )
    return helper.make_model(
        graph, opset_imports=[helper.make_opsetid('', 13)],
        ir_version=8)


def _build_variables():
    return {
        'inputs': [{
            'name': 'input',
            'variables': [
                {'name': 'v{}'.format(i), 'scale': 1.0, 'offset': 0.0}
                for i in range(LAYER_SIZES[0])
            ],
        }],
        'input_sequences': [],
        'outputs': [{
            'name': 'output',
            'labels': ['out_{}'.format(i) for i in range(LAYER_SIZES[-1])],
        }],
    }


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: {} <output-directory>".format(sys.argv[0]))
    out_dir = sys.argv[1]
    os.makedirs(out_dir, exist_ok=True)
    save_model(_build_model(), os.path.join(out_dir, 'dense.onnx'))
    with open(os.path.join(out_dir, 'variables.json'), 'w') as f:
        json.dump(_build_variables(), f, indent=2)


if __name__ == '__main__':
    main()
