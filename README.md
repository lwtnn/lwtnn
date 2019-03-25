Lightweight Trained Neural Network
==================================

[![Build Status][build-img]][build-link] [![Scan Status][scan-img]][scan-link]
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.597221.svg)](https://doi.org/10.5281/zenodo.597221)

[build-img]: https://travis-ci.org/lwtnn/lwtnn.svg?branch=master
[build-link]: https://travis-ci.org/lwtnn/lwtnn
[scan-img]: https://scan.coverity.com/projects/9285/badge.svg
[scan-link]: https://scan.coverity.com/projects/lwtnn-lwtnn

What is this?
-------------

The code comes in two parts:

 1. A set of scripts to convert saved neural networks to a standard
    JSON format
 2. A set of classes which reconstruct the neural network for
    application in a C++ production environment

The main design principles are:

 - **Minimal dependencies:** The C++ code depends on C++11,
   [Eigen][eg], and boost [PropertyTree][pt]. The converters have
   additional requirements (Python3 and h5py) but these can be run
   outside the C++ production environment.

 - **Easy to extend:** Should cover 95% of deep network architectures we
   would realistically consider.

 - **Hard to break:** The NN constructor checks the input NN for
   consistency and fails loudly if anything goes wrong.


We also include converters from several popular formats to the `lwtnn`
JSON format. Currently the following formats are supported:
 - Scikit Learn
 - [Keras][kr] (most popular, see below)

[eg]: http://eigen.tuxfamily.org
[pt]: http://www.boost.org/doc/libs/1_59_0/doc/html/property_tree.html
[kr]: http://keras.io/

Why are we doing this?
----------------------

Our underlying assumption is that _training_ and _inference_ happen in
very different environments: we assume that the training environment
is flexible enough to support modern and frequently-changing
libraries, and that the inference environment is much less flexible.

If you have the flexibility to run any framework in your production
environment, this package is _not_ for you. If you want to apply a
network you've trained with Keras in a 6M line C++ production
framework that's only updated twice a year, you'll find this package
very useful.

Getting the code
----------------

Clone the project from github:

```bash
git clone git@github.com:lwtnn/lwtnn.git
```

Then compile with `make`. If you have access to a relatively new
version of Eigen and Boost everything should work without errors.

If you have CMake, you can build with _no_ other dependencies:

```bash
mkdir build
cmake -DBUILTIN_BOOST=true -DBUILTIN_EIGEN=true ..
make -j 4
```

Running a full-chain test
-------------------------

If you have Python 3 and h5py installed you can run a test. Starting
from the directory where you built the project, run

```
./tests/test-GRU.sh
```

(note that if you ran `cmake` this is `../tests/test-GRU.sh`)

You should see some printouts that end with ` *** Success! *** `.

Quick Start With Keras Functional API
-------------------------------------

The following instructions apply to the model/functional API in
Keras. To see the instructions relevant to the sequential API, go to
[Quick Start With sequential API][seqQuickStart].

After building, there are some required steps:

##### 1) Save your network output file

Make sure you saved your architecture and weights file from Keras, and
created your input variable file. See [the lwtnn Keras Converter wiki
page][weightsInputs] for the _correct_ procedure in doing all of this.

Then

```
lwtnn/converters/kerasfunc2json.py architecture.json weights.h5 inputs.json > neural_net.json
```

<sup>Helpful hint: if you do `lwtnn/converters/kerasfunc2json.py architecture.json weights.h5` it creates a skeleton of an input file for you, which can be used in the above command!</sup>

##### 2) Test your saved output file

A good idea is to test your converted network:

```
./lwtnn-test-lightweight-graph neural_net.json
```

A basic regression test is performed with a bunch of random
numbers. This test just ensures that lwtnn can in fact read your NN.

[weightsInputs]: https://github.com/lwtnn/lwtnn/wiki/Keras-Converter
[seqQuickStart]: https://github.com/lwtnn/lwtnn/wiki/Quick-Start-With-Sequential-API

##### 3) Apply your saved neural network within C++ code

```C++
// Include several headers. See the files for more documentation.
// First include the class that does the computation
#include "lwtnn/LightweightGraph.hh"
// Then include the json parsing functions
#include "lwtnn/parse_json.hh"

...

// get your saved JSON file as an std::istream object
std::ifstream input("path-to-file.json");
// build the graph
LightweightGraph graph(parse_json_graph(input));

...

// fill a map of input nodes
std::map<std::string, std::map<std::string, double> > inputs;
inputs["input_node"] = {{"value", value}, {"value_2", value_2}};
inputs["another_input_node"] = {{"another_value", another_value}};
// compute the output values
std::map<std::string, double> outputs = graph.compute(inputs);
```

After the constructor for the class `LightweightNeuralNetwork` is
constructed, it has one method, `compute`, which takes a `map<string,
double>` as an input and returns a `map` of named outputs (of the same
type). It's fine to give `compute` a map with more arguments than the
NN requires, but if some argument is _missing_ it will throw an
`NNEvaluationException`.

All inputs and outputs are stored in `std::map`s to prevent bugs with
incorrectly ordered inputs and outputs. The strings used as keys in
the map are specified by the network configuration.


### Supported Layers ###

In particular, the following layers are supported as implemented in the
Keras sequential and functional models:

|                 | K sequential | K functional  |
|-----------------|--------------|---------------|
| Dense           |  yes         |  yes          |
| Normalization   | See Note 1   | See Note 1    |
| Maxout          |  yes         |  yes          |
| Highway         |  yes         |  yes          |
| LSTM            |  yes         |  yes          |
| GRU             |  yes         |  yes          |
| Embedding       | sorta        | [issue][ghie] |
| Concatenate     |  no          |  yes          |
| TimeDistributed |  no          |  yes          |

**Note 1:** Normalization layers (i.e. Batch Normalization) are only
supported for Keras 1.0.8 and higher.

[ghie]: https://github.com/lwtnn/lwtnn/issues/39
[ghkeras2]: https://github.com/lwtnn/lwtnn/issues/40

#### Supported Activation Functions ####

| Function      | Implemented? |
|---------------|--------------|
| ReLU          | Yes          |
| Sigmoid       | Yes          |
| Hard Sigmoid  | Yes          |
| Tanh          | Yes          |
| Softmax       | Yes          |
| ELU           | Yes          |
| LeakyReLU     | Yes          |
| Swish         | Yes          |

The converter scripts can be found in `converters/`. Run them with
`-h` for more information.

Have problems?
--------------

For more in-depth documentation please see the [`lwtnn` wiki][lwtnnwiki].

If you find a bug in this code, or have any ideas, criticisms,
etc, please email me at `dguest@cern.ch`.

[lwtnnwiki]: https://github.com/lwtnn/lwtnn/wiki
