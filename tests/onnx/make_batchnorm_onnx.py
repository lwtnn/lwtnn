#!/usr/bin/env python3
# Fixture generator for test-onnx-batchnorm.sh.
#
# Two-branch graph exercising BatchNormalization, Concat, and a
# graph-level Add merge:
#
#   input_a -> Gemm -> BatchNorm -> Relu \
#                                         Concat -> Gemm -> Add -> output
#   input_b -> Gemm -> BatchNorm -> Relu /                   ^
#                                                            |
#                       (input_b -> Gemm) ---------------> extra
import json
import os
import sys

import numpy as np
from onnx import TensorProto, helper, numpy_helper, save_model


SEED = 7
IN_A = 3
IN_B = 4
HID_A = 5
HID_B = 5
OUT = 3


def _gemm(name, x, n_in, n_out, rng, inits):
    w = rng.standard_normal((n_in, n_out)).astype(np.float32)
    b = rng.standard_normal((n_out,)).astype(np.float32)
    w_n, b_n = '{}_W'.format(name), '{}_B'.format(name)
    inits += [numpy_helper.from_array(w, w_n), numpy_helper.from_array(b, b_n)]
    out = '{}_out'.format(name)
    return helper.make_node('Gemm', [x, w_n, b_n], [out], name=name), out


def _bn(name, x, n, rng, inits):
    scale = rng.standard_normal((n,)).astype(np.float32)
    beta = rng.standard_normal((n,)).astype(np.float32)
    mean = rng.standard_normal((n,)).astype(np.float32)
    var = (rng.standard_normal((n,)).astype(np.float32) ** 2 + 0.1)
    names = ['{}_{}'.format(name, s) for s in ('scale', 'beta', 'mean', 'var')]
    inits += [
        numpy_helper.from_array(scale, names[0]),
        numpy_helper.from_array(beta, names[1]),
        numpy_helper.from_array(mean, names[2]),
        numpy_helper.from_array(var, names[3]),
    ]
    out = '{}_out'.format(name)
    return helper.make_node(
        'BatchNormalization', [x] + names, [out],
        name=name, epsilon=1e-5), out


def _build_model():
    rng = np.random.default_rng(SEED)
    inits = []
    nodes = []

    n, x = _gemm('gemm_a', 'input_a', IN_A, HID_A, rng, inits)
    nodes.append(n)
    n, x = _bn('bn_a', x, HID_A, rng, inits)
    nodes.append(n)
    nodes.append(helper.make_node('Relu', [x], ['relu_a'], name='relu_a'))

    n, y = _gemm('gemm_b', 'input_b', IN_B, HID_B, rng, inits)
    nodes.append(n)
    n, y = _bn('bn_b', y, HID_B, rng, inits)
    nodes.append(n)
    nodes.append(helper.make_node('Relu', [y], ['relu_b'], name='relu_b'))

    nodes.append(helper.make_node(
        'Concat', ['relu_a', 'relu_b'], ['concat'],
        name='concat', axis=-1))
    n, gemm_out = _gemm(
        'gemm_final', 'concat', HID_A + HID_B, OUT, rng, inits)
    nodes.append(n)

    n, extra = _gemm('gemm_extra', 'input_b', IN_B, OUT, rng, inits)
    nodes.append(n)
    nodes.append(helper.make_node(
        'Add', [gemm_out, extra], ['output'], name='add_merge'))

    graph = helper.make_graph(
        nodes, 'bn_test',
        inputs=[
            helper.make_tensor_value_info(
                'input_a', TensorProto.FLOAT, ['batch', IN_A]),
            helper.make_tensor_value_info(
                'input_b', TensorProto.FLOAT, ['batch', IN_B]),
        ],
        outputs=[helper.make_tensor_value_info(
            'output', TensorProto.FLOAT, ['batch', OUT])],
        initializer=inits,
    )
    return helper.make_model(
        graph, opset_imports=[helper.make_opsetid('', 13)],
        ir_version=8)


def _build_variables():
    return {
        'inputs': [
            {
                'name': 'input_a',
                'variables': [
                    {'name': 'a{}'.format(i), 'scale': 1.0, 'offset': 0.0}
                    for i in range(IN_A)
                ],
            },
            {
                'name': 'input_b',
                'variables': [
                    {'name': 'b{}'.format(i), 'scale': 1.0, 'offset': 0.0}
                    for i in range(IN_B)
                ],
            },
        ],
        'input_sequences': [],
        'outputs': [{
            'name': 'output',
            'labels': ['out_{}'.format(i) for i in range(OUT)],
        }],
    }


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: {} <output-directory>".format(sys.argv[0]))
    out_dir = sys.argv[1]
    os.makedirs(out_dir, exist_ok=True)
    save_model(_build_model(), os.path.join(out_dir, 'bn.onnx'))
    with open(os.path.join(out_dir, 'variables.json'), 'w') as f:
        json.dump(_build_variables(), f, indent=2)


if __name__ == '__main__':
    main()
