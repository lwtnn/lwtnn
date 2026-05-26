#!/usr/bin/env python3
#
# Converter from ONNX NN to JSON
"""____________________________________________________________________
Variable specification file

In addition to the ONNX model file, you must provide a "variable
specification" json file with the following format:

  {
    "inputs": [
      {"name": "input_node_name",
       "variables": [
         {"name": "var_0", "scale": 1, "offset": 0, "default": 0.0},
         ...
       ]},
      ...
    ],
    "input_sequences": [],
    "outputs": [
      {"name": "output_node_name", "labels": ["class_0", "class_1", ...]}
    ]
  }

Where `scale` and `offset` account for any scaling/shifting applied to
inputs before feeding the network. The "default" value is optional.

If no variables file is given, a template is printed to stdout. Pipe it
to a file, edit, then re-run with the template as the variables file.

Currently supported ops: Gemm, MatMul + Add (fused), BatchNormalization,
Concat, Add (merge), Relu, Sigmoid, Tanh, Softmax, HardSigmoid,
LeakyRelu, Elu, and the SiLU/Swish pattern Mul(x, Sigmoid(x)).
Shape-only ops (Identity, Cast, Dropout, Flatten, Reshape, Squeeze,
Unsqueeze) are dropped.
"""

import argparse
import json
import sys

import onnx
from onnx import numpy_helper

from onnx_converter_common import skip_ops, MIN_OPSET
from onnx_op_converters import op_converters


def _run():
    args = _get_args()
    model = onnx.load(args.model_file)
    _check_opset(model)
    initializers = {init.name: numpy_helper.to_array(init)
                    for init in model.graph.initializer}

    if not args.variables_file:
        _build_variables_file(model, initializers)
        sys.exit(0)

    with open(args.variables_file) as vf:
        variables = json.load(vf)

    layers, nodes, input_nodes, output_nodes = _build_graph(model, initializers)

    out_dict = {
        'layers': layers,
        'nodes': nodes,
        'inputs': _parse_inputs(variables['inputs'], input_nodes, dims=1),
        'input_sequences': _parse_inputs(
            variables.get('input_sequences', []), input_nodes, dims=2),
        'outputs': _parse_outputs(variables['outputs'], output_nodes),
    }
    print(json.dumps(out_dict, indent=2, sort_keys=True))


def _get_args():
    parser = argparse.ArgumentParser(
        description="Converter from ONNX saved NN to JSON",
        epilog=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('model_file', help='ONNX model file (.onnx)')
    parser.add_argument('variables_file', help='variable spec as json',
                        nargs='?')
    return parser.parse_args()


def _check_opset(model):
    for opset in model.opset_import:
        if opset.domain in ('', 'ai.onnx') and opset.version < MIN_OPSET:
            sys.exit(
                "ONNX opset {} is below the minimum supported opset {}. "
                "Re-export the model with a newer opset (e.g. "
                "`torch.onnx.export(..., opset_version={})`).".format(
                    opset.version, MIN_OPSET, MIN_OPSET))


# __________________________________________________________________________
# Graph construction

class Node:
    def __init__(self, kind, name, sources, n_outputs=None, dims=None,
                 onnx_node=None):
        self.kind = kind
        self.name = name
        self.sources = sources
        self.n_outputs = n_outputs
        self.dims = dims
        self.onnx_node = onnx_node
        self.number = None
        self.layer_number = None

    def op_type(self):
        return self.onnx_node.op_type if self.onnx_node else 'Input'


def _build_graph(model, initializers):
    tensor_to_node = {}

    input_nodes = {}
    for vi in model.graph.input:
        if vi.name in initializers:
            continue
        dims, n_features = _input_dims(vi)
        node = Node('input', vi.name, sources=[],
                    n_outputs=n_features, dims=dims)
        input_nodes[vi.name] = node
        tensor_to_node[vi.name] = node

    matmul_to_add = _find_matmul_add_fusions(model, initializers)
    fused_add_outputs = {add.output[0] for add in matmul_to_add.values()}

    sigmoid_to_silu_mul = _find_silu_fusions(model)
    silu_mul_outputs = {mul.output[0] for mul in sigmoid_to_silu_mul.values()}

    nodes_in_order = list(input_nodes.values())
    layers = []

    for onnx_node in model.graph.node:
        out_name = onnx_node.output[0]

        if onnx_node.op_type in skip_ops:
            src = _first_value_input(onnx_node, initializers, tensor_to_node)
            tensor_to_node[out_name] = tensor_to_node[src]
            continue

        if out_name in fused_add_outputs or out_name in silu_mul_outputs:
            continue

        sources = [tensor_to_node[t] for t in onnx_node.input
                   if t not in initializers and t in tensor_to_node]

        if onnx_node.op_type == 'Concat':
            n_out = sum(s.n_outputs for s in sources)
            node = Node('merge', out_name, sources, n_outputs=n_out,
                        onnx_node=onnx_node)
        elif onnx_node.op_type == 'Add' and len(sources) == 2:
            # two value inputs means a real merge, not a bias-add
            assert sources[0].n_outputs == sources[1].n_outputs, (
                "Add '{}' inputs have different widths ({} vs {})".format(
                    onnx_node.name, sources[0].n_outputs,
                    sources[1].n_outputs))
            node = Node('merge', out_name, sources,
                        n_outputs=sources[0].n_outputs,
                        onnx_node=onnx_node)
        else:
            if onnx_node.op_type not in op_converters:
                sys.exit(
                    "Unsupported ONNX op '{}' (node '{}'). Supported ops: "
                    "{}".format(
                        onnx_node.op_type, onnx_node.name,
                        ', '.join(sorted(op_converters))))
            assert len(sources) == 1, (
                "expected one value input for {} '{}', got {}".format(
                    onnx_node.op_type, onnx_node.name, len(sources)))
            n_in = sources[0].n_outputs
            extra = None
            if onnx_node.op_type == 'MatMul':
                extra = matmul_to_add.get(out_name)
                if extra is None:
                    sys.exit(
                        "MatMul '{}' (output '{}') is not followed by a "
                        "constant-bias Add. lwtnn requires Dense layers to "
                        "have a bias.".format(onnx_node.name, out_name))
                out_name = extra.output[0]
            if onnx_node.op_type == 'Sigmoid' and out_name in sigmoid_to_silu_mul:
                mul = sigmoid_to_silu_mul[out_name]
                layer_dict = {
                    'weights': [], 'bias': [], 'architecture': 'dense',
                    'activation': {'function': 'swish', 'alpha': 1.0},
                }
                n_out = n_in
                out_name = mul.output[0]
            else:
                layer_dict, n_out = op_converters[onnx_node.op_type](
                    initializers, onnx_node, n_in, extra=extra)
            layer_number = len(layers)
            layers.append(layer_dict)
            node = Node('op', out_name, sources, n_outputs=n_out,
                        onnx_node=onnx_node)
            node.layer_number = layer_number

        tensor_to_node[out_name] = node
        nodes_in_order.append(node)

    for n, node in enumerate(nodes_in_order):
        node.number = n

    output_nodes = {}
    for vi in model.graph.output:
        if vi.name not in tensor_to_node:
            sys.exit("Graph output '{}' is not produced by any node".format(
                vi.name))
        output_nodes[vi.name] = tensor_to_node[vi.name]

    node_list = _build_node_list(nodes_in_order, input_nodes)
    return layers, node_list, input_nodes, output_nodes


def _input_dims(value_info):
    """Return (rank-minus-batch, feature-count) for a graph input.

    A flat (batch, features) input gives dims=1; a sequence
    (batch, time, features) gives dims=2."""
    shape = value_info.type.tensor_type.shape.dim
    rank = len(shape) - 1
    n_features = shape[-1].dim_value
    if n_features == 0:
        sys.exit("Input '{}' has unknown feature dimension; ONNX models for "
                 "lwtnn must have a concrete feature size.".format(
                     value_info.name))
    return rank, n_features


def _first_value_input(onnx_node, initializers, tensor_to_node):
    values = [t for t in onnx_node.input
              if t not in initializers and t in tensor_to_node]
    assert len(values) == 1, (
        "skip op '{}' expected one value input, got {}".format(
            onnx_node.name, len(values)))
    return values[0]


def _find_silu_fusions(model):
    """Map Sigmoid output tensor -> downstream Mul NodeProto, for every
    SiLU pattern: a Mul that multiplies tensor X with Sigmoid(X). ONNX
    has no native SiLU op so SiLU/Swish appears in this form."""
    consumers = {}
    for n in model.graph.node:
        for inp in n.input:
            consumers.setdefault(inp, []).append(n)
    graph_outputs = {vo.name for vo in model.graph.output}

    sigmoid_input = {n.output[0]: n.input[0] for n in model.graph.node
                     if n.op_type == 'Sigmoid'}
    fusions = {}
    for n in model.graph.node:
        if n.op_type != 'Mul' or len(n.input) != 2:
            continue
        a, b = n.input
        for cand, other in ((a, b), (b, a)):
            if cand not in sigmoid_input:
                continue
            if sigmoid_input[cand] != other:
                continue
            # the Sigmoid output must not be consumed by anything else
            # or appear as a graph output, otherwise folding it would
            # lose a needed tensor
            if cand in graph_outputs:
                continue
            if len(consumers.get(cand, [])) != 1:
                continue
            fusions[cand] = n
            break
    return fusions


def _find_matmul_add_fusions(model, initializers):
    """Map MatMul output tensor -> downstream Add NodeProto, for every Add
    whose other input is a constant. torch.onnx.export emits this pattern
    in place of Gemm."""
    matmul_outputs = {n.output[0] for n in model.graph.node
                      if n.op_type == 'MatMul'}
    fusions = {}
    for n in model.graph.node:
        if n.op_type != 'Add' or len(n.input) != 2:
            continue
        constant_inputs = [i for i in n.input if i in initializers]
        value_inputs = [i for i in n.input if i not in initializers]
        if len(constant_inputs) == 1 and len(value_inputs) == 1:
            if value_inputs[0] in matmul_outputs:
                fusions[value_inputs[0]] = n
    return fusions


_NODE_TYPE_MAP = {
    'Input': 'input',
    'Gemm': 'feed_forward',
    'MatMul': 'feed_forward',
    'BatchNormalization': 'feed_forward',
    'Relu': 'feed_forward',
    'Sigmoid': 'feed_forward',
    'Tanh': 'feed_forward',
    'Softmax': 'feed_forward',
    'HardSigmoid': 'feed_forward',
    'LeakyRelu': 'feed_forward',
    'Elu': 'feed_forward',
    'Concat': 'concatenate',
    'Add': 'add',
}


def _build_node_list(nodes_in_order, input_nodes):
    input_ordering = {name: i for i, name in enumerate(input_nodes)}
    node_list = []
    for node in nodes_in_order:
        if node.kind == 'input':
            entry = {
                'type': 'input_sequence' if node.dims > 1 else 'input',
                'sources': [input_ordering[node.name]],
                'size': node.n_outputs,
            }
        else:
            node_type = _NODE_TYPE_MAP[node.op_type()]
            entry = {
                'type': node_type,
                'sources': [s.number for s in node.sources],
            }
            if node.layer_number is not None:
                entry['layer_index'] = node.layer_number
        node_list.append(entry)
    return node_list


# __________________________________________________________________________
# Variables template

def _build_variables_file(model, initializers):
    inputs = []
    input_seqs = []
    for vi in model.graph.input:
        if vi.name in initializers:
            continue
        dims, n_features = _input_dims(vi)
        entry = {
            'name': vi.name,
            'variables': [
                {'name': 'variable_{}'.format(i), 'scale': 1, 'offset': 0}
                for i in range(n_features)
            ],
        }
        (inputs if dims == 1 else input_seqs).append(entry)

    outputs = []
    for vo in model.graph.output:
        shape = vo.type.tensor_type.shape.dim
        n_out = shape[-1].dim_value if shape else 0
        outputs.append({
            'name': vo.name,
            'labels': ['out_{}'.format(i) for i in range(n_out)],
        })

    template = {
        'inputs': inputs,
        'input_sequences': input_seqs,
        'outputs': outputs,
    }
    print(json.dumps(template, indent=2, sort_keys=True))


def _parse_inputs(spec_list, input_nodes, dims):
    matching_by_name = {n.name: n for n in input_nodes.values()
                        if n.dims == dims}
    out = []
    for entry in spec_list:
        name = entry['name']
        if name not in matching_by_name:
            sys.exit("variable spec references unknown input '{}' "
                     "(known: {})".format(name, list(matching_by_name)))
        node = matching_by_name[name]
        assert len(entry['variables']) == node.n_outputs, (
            "input '{}' has {} ONNX features but variable spec lists "
            "{}".format(name, node.n_outputs, len(entry['variables'])))
        variables = []
        for v in entry['variables']:
            info = {k: v[k] for k in ('name', 'scale', 'offset')}
            if 'default' in v:
                info['default'] = v['default']
            variables.append(info)
        out.append({'name': name, 'variables': variables})
    return out


def _parse_outputs(spec_list, output_nodes):
    outputs = {}
    for entry in spec_list:
        name = entry['name']
        if name not in output_nodes:
            sys.exit("variable spec references unknown output '{}' "
                     "(known: {})".format(name, list(output_nodes)))
        node = output_nodes[name]
        assert len(entry['labels']) == node.n_outputs, (
            "output '{}' produces {} values but variable spec lists {} "
            "labels".format(name, node.n_outputs, len(entry['labels'])))
        outputs[name] = {
            'node_index': node.number,
            'labels': entry['labels'],
        }
    return outputs


if __name__ == '__main__':
    _run()
