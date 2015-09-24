What the hell is this?
----------------------

This is a few lightweight classes to apply a trained neural net. The
main design principles are:

 - **Minimal dependencies:** It should only depend on C++11 and Eigen
   (maybe I'll add boost to handle parsing of the configuration).
 - **Flat structure:** Each layer in the NN inherits from the `ILayer`
   abstract base class, the NN itself is just a stack of `ILayer`
   derived classes.
 - **Easy to extend:** Should cover 95% of deep network architectures we
   would realistically consider.

How do I use it?
----------------

#### Quick Start ####

Just run `./test-nn.sh`. If nothing goes wrong you should see
something like:

```
out1 4
out2 3
out3 2
out4 1
```

There may be some problems if you don't have the right version of
python or don't have `pyyaml` installed.

#### Cool, what the hell did that do? ####

Take a look inside `test-nn.sh`, it does two things:

 - Runs `./converters/agile2json.py`. This should write a JSON file to
   standard out.
 - Pipes the output to `./bin/lwtag-test-full`. This will construct a
   NN from the resulting JSON and run a single test pattern.

Of course this isn't very useful, to do more you have to understand...

#### The High Level Interface ####

Open `include/LWTagger.hh` and find the class declaration for `LWTagger`. The constructor takes three arguments:

 - A vector of `Input`s: these structures give the variable `name`,
   `offset`, and `scale`. Note that these are applied as `v = (input +
   offset) * scale`, so if you're normalizing inputs with some `mean`
   and `standard_deviation`, these are given by `offset = - mean` and
   `scale = 1 / standard_deviation`.
 - A vector of `LayerConfig` structures. See the below section for an
   explanation of this class.
 - A vector of output names.

The constructor should check to make sure everything makes sense
internally. If anything goes wrong it will throw an exception.

After the class is constructed, it has one method, `compute`, which
takes a `map` of named doubles as an input and returns a `map` of
named outputs.

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

To Do List
----------

 - The copy and assignment constructors for `LWTagger` and `Stack` are
   currently deleted, because the defaults would cause all kinds of
   problems and I'm too lazy to write custom versions. It's not clear
   that we'll need them anyway, but if someone ends up wanting
   something like a `std::map<XXX, LWTagger>` I could add them.
