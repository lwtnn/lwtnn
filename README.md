What is this?
-------------

This is a few lightweight classes to apply a trained neural net. The
main design principles are:

 - **Minimal dependencies:** The core class should only depend on
   C++11 and [Eigen][eg]. The JSON parser to read in NNs also requires
   boost [PropertyTree][pt].
 - **Flat structure:** Each layer in the NN inherits from the `ILayer`
   or `IRecurrentLayer` abstract base class, the NN itself is just a
   stack of derived classes.
 - **Easy to extend:** Should cover 95% of deep network architectures we
   would realistically consider.
 - **Hard to break:** The NN constructor checks the serialized NN for
   consistency. To prevent bugs with incorrectly ordered variables,
   all inputs and outputs are stored in `std::map`s.

We also include converters from several popular formats to the `lwtnn` JSON format. Currently the following formats are supported:
 - [AGILEPack][ap]
 - [Keras][kr] (currently supports Dense, LSTM, and GRU, working on
   Maxout and Highway layers)
 - [Julian's][julian] format, based on Numpy and JSON

The converter scripts can be found in `converters/`. Run them with
`-h` for more information.

How do I use it?
----------------

#### Quick Start ####

After running `make`, just run `./tests/test-agilepack.sh`. If nothing
goes wrong you should see something like:

```
Running conversion ./convert/agile2json.py data/agile-in.yml
Testing with ./bin/lwtnn-test-arbitrary-net
all outputs within thresholds!
cleaning up
```

There may be some problems if you don't have python 3 or don't have
[`pyyaml`][pyy] installed, but these should be limited to the YAML ->
JSON converter. At the very least calling `./bin/lwtag-test-hlwrapper`
with no arguments (which doesn't depend on the converter) should work.

#### Cool, what the hell did that do? ####

Take a look inside the test routine, it does a few things:

 - Runs `./converters/agile2json.py`. This takes an [AGILEPack][ap]
   output and write a JSON file to standard out.
 - Sends the output to `./bin/lwtag-test-arbitrary-net`. This will
   construct a NN from the resulting JSON and run a single test
   pattern.

Of course this isn't very useful, to do more you have to understand...

Code Organization
-----------------

Code is intentionally organized into only a few files to make it
easier to copy into existing packages. The main files are:

 - `Stack` files: contain the low level NN classes, and any code
   that relies on Eigen.
 - `LightweightNeuralNetwork` files: contain the high-level wrappers,
   which implement STL (rather than Eigen) interfaces. To speed up
   compilation the header file can be included without including
   Eigen.
 - `NNLayerConfig` header: defines the structures to initialize networks.
 - `parse_json` files: contain functions to build the config
   structures from JSON.

There are a few other less important files that contain debugging code
and utilities.

#### The High Level Interface ####

Open `include/LightweightNeuralNetwork.hh` and find the class
declaration for `LightweightNeuralNetwork`. The constructor takes
three arguments:

 - A vector of `Input`s: these structures give the variable `name`,
   `offset`, and `scale`. Note that these are applied as `v = (input +
   offset) * scale`, so if you're normalizing inputs with some `mean`
   and `standard_deviation`, these are given by `offset = - mean` and
   `scale = 1 / standard_deviation`.
 - A vector of `LayerConfig` structures. See the below section for an
   explanation of this class.
 - A vector of output names.

The constructor should check to make sure everything makes sense
internally. If anything goes wrong it will throw a
`NNConfigurationException`.

After the class is constructed, it has one method, `compute`, which
takes a `map<string, double>` as an input and returns a `map` of named
outputs (of the same type). It's fine to give `compute` a map with
more arguments than the NN requires, but if some argument is _missing_
it will throw an `NNEvaluationException`. All the exceptions inherit
from `LightweightNNException`.

#### The Low Level Interface ####

The `Stack` class is initialized with two parameters: the number of
input parameters, and a `std::vector<LayerConfig>` to specify the
layers. Each `LayerConfig` structure contains:

 - A vector of `weights`. This can be zero-length, in which case no
   matrix is inserted (but the bias and activation layers are).
 - A `bias` vector. Again, it can be zero length for no bias in this
   layer.
 - An `activation` function. Defaults to `LINEAR` (i.e. no activation
   function).

Note that the dimensions of the matrices aren't specified after the
`n_inputs` in the `Stack` constructor, because this should be
constrained by the dimensions of the `weight` vectors. If something
doesn't make sense the constructor should throw an
`NNConfigurationException`.

The `Stack::compute(VectorXd)` method will return a `VectorXd` of
outputs.

Testing an Arbitrary NN
-----------------------

The `lwtnn-test-arbitrary-net` executable takes in a JSON file along
with two text files, one to specify the variable names and another to
give the input values. Run with no arguments to get help.

Recurrent Networks
------------------

Currently we support LSTMs in sequential models. The low level
interface is implemented as `RecurrentStack`. See `lwtnn-test-rnn` for
a working example.

Have problems?
--------------

If you find a bug in this code, or have any ideas, criticisms, etc, please email me at `dguest@cern.ch`.

[![Build Status][build-img]][build-link] [![Scan Status][scan-img]][scan-link]

To Do List
----------

 - The copy and assignment constructors for `LightweightNeuralNetwork`
   and `Stack` are currently deleted, because the defaults would cause
   all kinds of problems and I'm too lazy to write custom
   versions. It's not clear that we'll need them anyway, but if
   someone ends up wanting something like a `std::map<XXX,
   LightweightNeuralNetwork>` I could add them.


[ap]: https://github.com/lukedeo/AGILEPack
[kr]: http://keras.io/
[eg]: http://eigen.tuxfamily.org
[pt]: http://www.boost.org/doc/libs/1_59_0/doc/html/property_tree.html
[pyy]: http://pyyaml.org/wiki/PyYAML
[julian]: https://github.com/dguest/lw-client/wiki/Julian-file-format
[build-img]: https://travis-ci.org/dguest/lwtnn.svg?branch=master
[build-link]: https://travis-ci.org/dguest/lwtnn
[scan-img]: https://scan.coverity.com/projects/9285/badge.svg
[scan-link]: https://scan.coverity.com/projects/dguest-lwtnn
