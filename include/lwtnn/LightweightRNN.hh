#ifndef LIGHTWEIGHT_RNN_HH
#define LIGHTWEIGHT_RNN_HH

// Lightweight Recurrent NN
//
//basic code for forward pass computation of recurrent NN structures,
// like LSTM, useful for processing time series / sequence data.  goal
// to be able to evaluate Keras (keras.io) models in c++ in
// lightweight way
//
// Author: Michael Kagan <mkagan@cern.ch>

#include "NNLayerConfig.hh"
#include "LightweightNeuralNetwork.hh"

#include <Eigen/Dense>

#include <vector>

namespace lwt {

  typedef std::map<std::string, std::vector<double> > VectorMap;

  using Eigen::VectorXd;
  using Eigen::VectorXi;

  using Eigen::MatrixXd;

  //was going to use std::ptr_fun to reference function, which this
  //typedef may help with now using std::function, since ptr_fun is
  //deprecated, and will be removed in c++17 typedef double
  //(*activation_type)(double);


  class InputVectorPreprocessor
  {
  public:
    InputVectorPreprocessor(const std::vector<Input>& inputs);
    MatrixXd operator()(const VectorMap&) const;
  private:
    // input transformations
    VectorXd _offsets;
    VectorXd _scales;
    std::vector<std::string> _names;
  };


  class LightweightRNN
  {
  public:
    LightweightRNN(const std::vector<Input>& inputs,
                   const std::vector<LayerConfig>& layers,
                   const std::vector<std::string>& outputs);
    LightweightRNN(LightweightRNN&) = delete;
    LightweightRNN& operator=(LightweightRNN&) = delete;

    ValueMap reduce(const std::vector<ValueMap>&) const;
    ValueMap reduce(const VectorMap&) const;
  private:
    RecurrentStack _stack;
    InputPreprocessor _preproc;
    InputVectorPreprocessor _vec_preproc;
    std::vector<std::string> _outputs;
    size_t _n_inputs;
  };

}





#endif
