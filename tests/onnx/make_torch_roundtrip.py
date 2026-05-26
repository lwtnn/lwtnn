#!/usr/bin/env python3
# Fixture generator for test-torch-onnx-roundtrip.sh.
#
# Builds an MLP in PyTorch, exports to ONNX, computes the torch forward
# pass on a fixed input, and writes:
#   <out>/model.onnx
#   <out>/variables.json   lwtnn variable spec
#   <out>/inputs.json      input values for lwtnn-test-lightweight-graph
#   <out>/expected.json    torch forward pass, in reg-test.py --graph format
import json
import os
import sys

import torch
import torch.nn as nn


N_IN = 4
N_OUT = 3
SEED = 1234


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_IN, 8),
            nn.ReLU(),
            nn.Linear(8, 6),
            nn.Tanh(),
            nn.Linear(6, N_OUT),
            nn.Softmax(dim=-1),
        )

    def forward(self, x):
        return self.net(x)


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: {} <output-directory>".format(sys.argv[0]))
    out_dir = sys.argv[1]
    os.makedirs(out_dir, exist_ok=True)

    torch.manual_seed(SEED)
    model = MLP().eval()

    x = torch.linspace(-1.0, 1.0, N_IN).reshape(1, N_IN)
    with torch.no_grad():
        y = model(x).numpy().reshape(-1).tolist()

    onnx_path = os.path.join(out_dir, 'model.onnx')
    torch.onnx.export(
        model, x, onnx_path,
        input_names=['input'], output_names=['output'],
        dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}},
        opset_version=13,
    )

    variables = {
        'inputs': [{
            'name': 'input',
            'variables': [
                {'name': 'v{}'.format(i), 'scale': 1.0, 'offset': 0.0}
                for i in range(N_IN)
            ],
        }],
        'input_sequences': [],
        'outputs': [{
            'name': 'output',
            'labels': ['out_{}'.format(i) for i in range(N_OUT)],
        }],
    }
    with open(os.path.join(out_dir, 'variables.json'), 'w') as f:
        json.dump(variables, f, indent=2)

    inputs = {
        'input': {
            'v{}'.format(i): float(x[0, i].item()) for i in range(N_IN)
        }
    }
    with open(os.path.join(out_dir, 'inputs.json'), 'w') as f:
        json.dump(inputs, f, indent=2)

    # reg-test.py keys nodes by the literal `<name>:` line from
    # lwtnn-test-lightweight-graph, colon included
    expected = {
        'output:': {
            'out_{}'.format(i): [y[i]] for i in range(N_OUT)
        }
    }
    with open(os.path.join(out_dir, 'expected.json'), 'w') as f:
        json.dump(expected, f, indent=2)


if __name__ == '__main__':
    main()
