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

What's left to do?
------------------

 - The `Stack` class doesn't take care of arranging named inputs or
   outputs. Should add a `LWTagger` top level class to do this. This
   class can also handle transformation of inputs (can use something
   like a vector of `InputTransform` structures as a constructor
   argument).
 - We'll need a way to construct the NN from a text file. The plan is
   to write a function that produces a vector of `LayerConfig`, a
   vector of `InputTransform` and a vector of output names, which can
   be fed to the `LWTagger` constructor.
