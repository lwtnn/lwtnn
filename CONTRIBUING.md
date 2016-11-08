How to contribute
=================

First of all, thanks for your interest! We can use a lot of help with
this package and while writing code is awesome we also like to spend
some time applying it to our particle smasher data. Anything you can
do to help us find bugs, develop this code, or just provide feedback
is extremely helpful to us.

For common questions and (limited) documentation, please look at our
[github wiki][1]. If you find this inadequate we encourage you to
[create an issue][2]. And of course if you if you have any questions
at all, feel free to email the package maintainer at `dguest@cern.ch`.

Coding Standards
----------------

As a rule of thumb, no new dependencies are allowed in the C++ code.

### Version Control ###

 - Don't submit pull requests that add binary files or data
 - We follow a forking workflow: if you want to contribute fork the
   repository and submit a pull request

### General Syntax ###

 - Break lines at 80 characters
 - Never use tabs
 - Avoid trailing whitespace

### Python Converters ###

 - We use python 3
 - Follow [PEP 8][3]
 - Output should be a valid JSON NN (no "logging" info allowed)

### C++ ###

 - Generally follow K&R style (OTBS)
 - No print statements from central NN code
 - Errors throw exceptions inheriting from `LightweightNNException`

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

[1]: https://github.com/dguest/lwtnn/wiki
[2]: https://github.com/dguest/lwtnn/issues
[3]: https://www.python.org/dev/peps/pep-0008/
[4]: https://github.com/dguest/lwtnn-test-data
[5]: https://github.com/dguest/lwtnn/wiki/Testing-Your-NN
