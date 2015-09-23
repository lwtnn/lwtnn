#include <Python.h>

#include <stdexcept>

static PyObject* py_alg(PyObject *self, PyObject *args)
{
  const char* echo_this;
  int some_int = 0;
  bool ok = PyArg_ParseTuple(args,"s|i:alg", &echo_this, &some_int);
  if (!ok) return NULL;
  try {
    int newret = some_int;
    printf("hi %s %i\n", echo_this, newret);
    return Py_BuildValue("i", newret);
  }
  catch (const std::exception& e) {
    PyErr_SetString(PyExc_Exception, e.what());
    return NULL;
  }
}


static PyMethodDef methods[] = {
  {"test", py_alg, METH_VARARGS,
   "don't ask, read the source"},
  {NULL, NULL, 0, NULL}   /* sentinel */
};

static struct PyModuleDef lwtag = {
   PyModuleDef_HEAD_INIT,
   "testpy",   /* name of module */
   "this be testpy", /* module documentation, may be NULL */
   -1,       /* size of per-interpreter state of the module,
                or -1 if the module keeps state in global variables. */
   methods
};

extern "C" {
  PyMODINIT_FUNC PyInit_lwtag(void)
  {
    return PyModule_Create(&lwtag);
  }
}
