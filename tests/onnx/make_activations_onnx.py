#!/usr/bin/env python3
# Fixture generator for test-onnx-activations.sh.
#
# A single-branch graph covering the ops the dense/batchnorm fixtures
# don't already exercise:
#
#   Sigmoid, HardSigmoid, LeakyRelu (alpha != default),
#   Elu (alpha != default), MatMul + Add (fused dense),
#   the SiLU/Swish pattern (Mul of x with Sigmoid(x)),
#   and the skip ops Identity, Cast, Reshape, Unsqueeze, Squeeze, Dropout.
import json
import os
import sys

import numpy as np
from onnx import TensorProto, helper, numpy_helper, save_model


SEED = 99
N = 4
OUT = 3


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: {} <output-directory>".format(sys.argv[0]))
    out_dir = sys.argv[1]
    os.makedirs(out_dir, exist_ok=True)

    rng = np.random.default_rng(SEED)
    inits = []

    w1 = rng.standard_normal((N, N)).astype(np.float32)
    b1 = rng.standard_normal((N,)).astype(np.float32)
    w2 = rng.standard_normal((N, N)).astype(np.float32)
    b2 = rng.standard_normal((N,)).astype(np.float32)
    w3 = rng.standard_normal((N, OUT)).astype(np.float32)
    b3 = rng.standard_normal((OUT,)).astype(np.float32)

    for arr, name in [
        (w1, 'W1'), (b1, 'B1'),
        (w2, 'W2'), (b2, 'B2'),
        (w3, 'W3'), (b3, 'B3'),
    ]:
        inits.append(numpy_helper.from_array(arr, name))

    # shape and axes tensors for Reshape / Unsqueeze / Squeeze (opset 13+)
    reshape_shape = np.array([-1, N], dtype=np.int64)
    unsqueeze_axes = np.array([1], dtype=np.int64)
    squeeze_axes = np.array([1], dtype=np.int64)
    inits += [
        numpy_helper.from_array(reshape_shape, 'reshape_shape'),
        numpy_helper.from_array(unsqueeze_axes, 'unsqueeze_axes'),
        numpy_helper.from_array(squeeze_axes, 'squeeze_axes'),
    ]

    nodes = [
        helper.make_node('Identity', ['input'], ['t_id'], name='id'),
        helper.make_node('Gemm', ['t_id', 'W1', 'B1'], ['t_g1'], name='g1'),
        helper.make_node('Sigmoid', ['t_g1'], ['t_sig'], name='sig'),
        helper.make_node(
            'Reshape', ['t_sig', 'reshape_shape'], ['t_rs'], name='rs'),
        helper.make_node(
            'Cast', ['t_rs'], ['t_cast'], name='cast', to=TensorProto.FLOAT),
        helper.make_node(
            'Unsqueeze', ['t_cast', 'unsqueeze_axes'], ['t_un'], name='un'),
        helper.make_node(
            'Squeeze', ['t_un', 'squeeze_axes'], ['t_sq'], name='sq'),
        helper.make_node('MatMul', ['t_sq', 'W2'], ['t_mm'], name='mm'),
        helper.make_node('Add', ['t_mm', 'B2'], ['t_dense'], name='mm_add'),
        helper.make_node(
            'LeakyRelu', ['t_dense'], ['t_lr'], name='lr', alpha=0.2),
        helper.make_node('Dropout', ['t_lr'], ['t_drop'], name='drop'),
        helper.make_node('Gemm', ['t_drop', 'W3', 'B3'], ['t_g3'], name='g3'),
        helper.make_node('Elu', ['t_g3'], ['t_elu'], name='elu', alpha=2.0),
        # SiLU/Swish, written as Mul(x, Sigmoid(x))
        helper.make_node('Sigmoid', ['t_elu'], ['t_silu_sig'], name='silu_sig'),
        helper.make_node(
            'Mul', ['t_elu', 't_silu_sig'], ['t_silu'], name='silu_mul'),
        helper.make_node('HardSigmoid', ['t_silu'], ['output'], name='hs'),
    ]

    graph = helper.make_graph(
        nodes, 'activations_test',
        inputs=[helper.make_tensor_value_info(
            'input', TensorProto.FLOAT, ['batch', N])],
        outputs=[helper.make_tensor_value_info(
            'output', TensorProto.FLOAT, ['batch', OUT])],
        initializer=inits,
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid('', 13)],
        ir_version=8)
    save_model(model, os.path.join(out_dir, 'activations.onnx'))

    variables = {
        'inputs': [{
            'name': 'input',
            'variables': [
                {'name': 'v{}'.format(i), 'scale': 1.0, 'offset': 0.0}
                for i in range(N)
            ],
        }],
        'input_sequences': [],
        'outputs': [{
            'name': 'output',
            'labels': ['out_{}'.format(i) for i in range(OUT)],
        }],
    }
    with open(os.path.join(out_dir, 'variables.json'), 'w') as f:
        json.dump(variables, f, indent=2)


if __name__ == '__main__':
    main()
