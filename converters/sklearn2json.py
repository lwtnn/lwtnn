"""
Created:      1  December 2017
Last Updated: 12 December 2017

Dan Marley
daniel.edison.marley@cernSPAMNOT.ch
Texas A&M University

Justin Pilot
UC Davis

---

Scikit-learn model -> lwtnn converter

Convert simple sklearn MLP Classifer to proper format for lwtnn framework.
Generate JSON file with variables (inputs to NN) with offset and scale values.
  Based on converters/keras2json.py:   https://github.com/lwtnn/lwtnn


WARNING: Some values are hard-coded due to my unfamiliarity
         with the sklearn MLPClassifier interface
         http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html

To run and show list of options:
   python converters/sklearn2json.py --help

NB:
   scale/offset defined from "StandardScaler"
     http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
     :: offset = -mean                   (StandardScaler.mean_)
     :: scale  = 1 / standard_deviation  (1/StandardScaler.scale_)
"""
import sys
import json
import numpy as np
from argparse import ArgumentParser
from sklearn import svm, metrics, preprocessing
import joblib


class Sklearn2json(object):
    """Class to convert scikit-learn model to JSON format for lwtnn"""
    def __init__(self):
        """Set attributes of the class"""
        # converting model
        self.model          = None          # model from sklearn
        self.scaler_file    = None          # StandardScaler object from sklearn
        self.output_file    = None          # name of output file to store JSON
        self.variables_file = None          # name of JSON file that contains variables

        # generate variables JSON
        self.makeVariablesJSON = False      # boolean to (not) make new variables JSON
        self.listOfVariables   = None       # name of text file that contains list of variables

        # parameters for converting model
        self.architecture      = "dense"    # always appears to be this (fully-connected MLPClassifier)
        self.nodeActivation    = "linear"   # always appears to be this
        self.class_labels      = None       # names in map for NN outputs
        self.miscellaneous     = None       # misc. info

        # extra parameters needed
        self.activation_fns    = {'relu':'rectified','softmax':'softmax','logistic':'sigmoid'} # sklearn->lwtnn naming

        # storing output (JSON)
        self.output = {"defaults":{},
                       "inputs":[],
                       "layers":[],
                       "miscellaneous":{},
                       "outputs":[]}


    def execute(self):
        """Run the converter"""
        print ( " Running scikit-learn to JSON converter " )
        if self.makeVariablesJSON:
            print ( f" > Generating variables JSON file {self.variables_file}" )
            self.generateVariablesJSON()  # generate the variables JSON file from scratch

        print ( " > Load sklearn model " )
        self.loadModel()       # load the sklearn model

        print ( " > Load variables JSON " )
        self.loadVariables()   # load the inputs (variables/scales/offsets/misc./class labels)

        print ( " > Load layers of neural network into dictionary " )
        self.loadLayers()      # load the network layers

        print ( " > Save model to {self.output_file}" )
        self.saveModel()       # save the model to JSON output

        return


    def loadModel(self):
        """Load the model (saved with joblib!)"""
        self.mlp = joblib.load(self.model)       # load model

        self.activation    = self.mlp.activation      # activation functions (one for all hidden layers)
        self.outActivation = self.mlp.out_activation_ # output activation
        self.weights       = self.mlp.coefs_          # weights vectors
        self.biases        = self.mlp.intercepts_     # bias vectors
        self.nLayers       = self.mlp.n_layers_       # number of layers in NN (input + N hidden + output)
        self.nOutputs      = self.mlp.n_outputs_      # number of output values
        self.sizeOfHLs     = self.mlp.hidden_layer_sizes   # tuple of hidden layer sizes
        self.nHiddenLayers = len(self.sizeOfHLs)      # number of hidden layers

        return


    def loadVariables(self):
        """Inputs (variables, weights + offsets).
           Read the variables JSON file to better understand what is being saved.
        """
        self.variables = json.load( open(self.variables_file) ) # contains variables, class_labels, and misc.

        self.output["inputs"]        = self.variables["inputs"]
        self.output["miscellaneous"] = self.variables["miscellaneous"]

        output_names = self.variables["class_labels"]
        if self.nOutputs!=len(output_names):
            print (" WARNING:  Number of outputs ({0}) "
                  "and number of output names ({1}) do not match!".format(self.nOutputs,output_names))
            print ( " WARNING:  Please check the model and list of output names." )
            sys.exit(1)

        self.output["outputs"] = output_names

        return


    def loadLayers(self):
        """
        Connecting layer (l) to layer (l+1) & storing in a dictionary (JSON):
           IN  -> Hidden Layer l (HL1)
           HL1 -> HL2
           ...
           HLN -> OUT
        """
        if any( [self.nLayers != len(elem)+1 for elem in [self.weights,self.biases] ] ):
            print (" WARNING:  Number of hidden layers ({0}) "
                  "and length of weights ({1}) do not match!".format(self.nLayers,self.weights+1))
            print ( " WARNING:  Please check the model." )
            sys.exit(1)

        for l in range(self.nLayers-1):
            # connecting layer (l) to layer (l+1)
            layer = {}
            layer["architecture"] = self.architecture
            layer["activation"]   = self.nodeActivation
            layer["weights"]      = self.weights[l].T.flatten().tolist()
            layer["bias"]         = self.biases[l].flatten().tolist()

            self.output["layers"].append(layer)

            # activation function at layer (l+1)
            activation_fn = self.activation if l!=self.nLayers-2 else self.outActivation
            activation_fn = self.activation_fns[activation_fn]
            act_fn = {
              "activation":   activation_fn,
              "architecture": self.architecture,
              "bias": [],    # always empty, I think (?)
              "weights": []  # always empty, I think (?)
            }
            self.output["layers"].append(act_fn)

        return


    def saveModel(self):
        """Save directly to JSON"""
        result = json.dumps(self.output, indent=2, sort_keys=True)

        f = open(self.output_file, 'w')
        f.write(result)
        f.close()

        return


    def generateVariablesJSON(self):
        """From list of variables & StandardScaler, generate variables JSON file"""
        scaler = joblib.load(self.scaler_file)
        vars   = open(self.listOfVariables,"r").readlines()

        variablesJSON = {"inputs": [],
                         "class_labels": self.class_labels,
                         "miscellaneous":self.miscellaneous}

        if scaler.n_features_in_ != len(vars):
            print (" WARNING:  Number of variables in file ({0}) "
                  "and number of scaler variables ({1}) do not match!".format(len(vars),scaler.n_features_in_))
            print ( " WARNING:  Please check the model and list of output names." )
            sys.exit(1)

        # generate inputs
        for i,var in enumerate(vars):
            var = var.rstrip('\n')
            input = {"name": var,
                     "offset":-scaler.mean_[i],
                     "scale":1/scaler.scale_[i] }    # scaler.scale_[i] = sqrt(scaler.var_[i])
            variablesJSON["inputs"].append(input)

        result = json.dumps(variablesJSON, indent=2, sort_keys=True)

        f = open(self.variables_file, 'w')
        f.write(result)
        f.close()

        return



# Convert scikit-learn model to JSON
if __name__ == '__main__':

    ## Parse arguments
    parser = ArgumentParser(description="Scikit-learn Converter")

    #  -- Making the lwtnn JSON file
    parser.add_argument('-m','--model', action='store',
                        default='mlp.pkl',
                        dest='model',
                        help='Name of the model from scikit-lean to convert.')
    parser.add_argument('-v','--variables', action='store',
                        default='mlp_variables.json',
                        dest='variables',
                        help='Name of JSON file that contains variables (with offsets and scales) in model.')
    parser.add_argument('-o','--output', action='store',
                        default='mlp.json',
                        dest='output',
                        help='Name of output file to save the JSON model.')
    parser.add_argument('-s','--scaler', action='store',
                        default='scaler.pkl',
                        dest='scaler',
                        help='Name of the scaler from scikit-learn to scale inputs.')

    # -- Making the variables JSON file
    parser.add_argument('-c','--class_labels',action='store',
                        default='output',
                        dest='class_labels',
                        help='Comma-separated names of NN output(s)')
    parser.add_argument('-l','--listOfVariables', action='store',
                        default='mlp_variables.txt',
                        dest='listOfVariables',
                        help='Name of TXT file that contains variables in model.')
    parser.add_argument('-mv','--make_variables', action='store_true',
                        dest='make_variables',
                        help='Make the JSON file that contains variable names with scaling and offset values.')

    results = parser.parse_args()



    ## Setup sklearn2json object
    conv = Sklearn2json()
    conv.model          = results.model          # scikit-learn model, saved with 'joblib'
    conv.scaler_file    = results.scaler         # scikit-learn scaler, saved with 'joblib'
    conv.output_file    = results.output         # output JSON file for lwtnn
    conv.variables_file = results.variables      # dictionary of variables, offsets, and scales

    # generate variables JSON (add options later, if needed)
    conv.makeVariablesJSON = results.make_variables
    if results.make_variables:
        conv.listOfVariables = results.listOfVariables         # list of variables used in NN
        conv.class_labels    = results.class_labels.split(',') # list of names for output values
        conv.miscellaneous   = {"scikit-learn": "0.18.1"}      # scikit-learn version

    # execute the converter!
    conv.execute()


## THE END
