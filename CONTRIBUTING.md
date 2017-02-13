How to contribute
=================

First of all, thanks for your interest! Anything you can do to provide
feedback is extremely helpful to us.

For common questions and documentation, please look at our
[github wiki][1]. If you find this inadequate we encourage you to
[create an issue][2]. And of course if you if you have any questions
at all, feel free to email the package maintainer at `dguest@cern.ch`.

Coding Standards
----------------

As a rule of thumb, no new dependencies are allowed in the C++ code.

### Submitting Pull Requests ###

 - We follow a forking workflow: if you want to contribute fork the
   repository and submit a pull request
 - Don't submit pull requests that add binary files or data
 - Write [meaningful commit messages][6]

### General Syntax ###

 - **Break lines at 80 characters.** It's easier to read and allows
   you to view two files side-by-side.
 - **Never use tabs.** Text editors always screw them up. Instead
   indent with multiple spaces.
 - Avoid trailing whitespace

### Python Converters ###

There are several scripts which convert common NN formats into JSON
which lwtnn can use to build networks. We're happy to add more formats
and extend the current converters, provided that you follow a few
standards:

 - Code should be written in python 3
 - Follow [PEP 8][3]
 - The converters should write JSON to stdout such that it can be
   piped into a file. This means that their only output should be a
   valid JSON NN (no "logging" info).

### C++ Classes ###

The code can be broken into several groups. The "core" classes are
divided into a high-level and low-level interface.

 - **Low-level core classes** in `Stack` supply bare Eigen interfaces
   and handles the implementation of the various layers.
 - **High-level interfaces** are in `LightweightNeuralNetwork`. These
   headers should _not_ include Eigen headers, and support interfaces
   via `std::map` and `std::vector`.

In addition there are several files for "peripheral" code:

 - **Configuration Handling** code includes that in `NNLayerConfig`
   and `parse_json`. It should only be necessary to build the NN
   objects and thus isn't performance critical.
 - **Test executables** must begin with `lwtnn-` (this is enforced by
   the makefile), and should be lower-case with `-`'s separating words

For consistency the **indentation should follow K&R style**
(OTBS). Also keep in mind the following guidelines:

 - **No print statements** outside test executables. Logging is
   potentially disastrous for execution times, adds useless noise, and
   is less effective for debugging than a set of unit tests
 - **Errors in core classes throw exceptions** and all exceptions
   should inherit from `LightweightNNException`. Please report any
   code that can segfault, produce undefined behavior, or throw
   standard library exceptions, we consider this a bug.



Reporting Bugs
--------------

If you think you found a bug please report it via [github][2].

Writing Unit Tests
------------------

If you implement a new layer, or just do something awesome with this
code, we encourage you to write a unit test around your NN. For
an example see `tests/test-ipmp.sh`. This simple script will:

 - **Download a Keras NN:** NN files are stored in a
   [separate repository][4] for now
 - Run `converters/keras2json.py` on the Keras files and pipe the
   output into a json file
 - Run a C++ test script that evaluates a dummy pattern. For more info
   see the [Testing Your NN][5] wiki.
 - Check the output against the expected output with `reg-test.py`

Any error in this process will cause the script to return a nonzero
exit code.

The unit tests are run automatically by Travis CI, and are listed in
the `script:` section of `.travis.yml`.

[1]: https://github.com/lwtnn/lwtnn/wiki
[2]: https://github.com/lwtnn/lwtnn/issues
[3]: https://www.python.org/dev/peps/pep-0008/
[4]: https://github.com/lwtnn/lwtnn-test-data
[5]: https://github.com/lwtnn/lwtnn/wiki/Testing-Your-NN
[6]: http://chris.beams.io/posts/git-commit/#seven-rules
