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

#include <Eigen/Dense>

#include <vector>
#include <functional>

#include "lwtnn/LightweightNeuralNetwork.hh"



namespace lwt {

  using Eigen::VectorXd;
  using Eigen::VectorXi;

  using Eigen::MatrixXd;

  //was going to use std::ptr_fun to reference function, which this
  //typedef may help with now using std::function, since ptr_fun is
  //deprecated, and will be removed in c++17 typedef double
  //(*activation_type)(double);

  // forward declare activation functions
  double nn_sigmoid( double x );
  double nn_hard_sigmoid( double x );
  double nn_tanh( double x );

  /// base recurrent class ///
  class IRecurrentLayer
  {
  public:
    virtual ~IRecurrentLayer() {}
    virtual MatrixXd scan( const MatrixXd&) = 0;
  };

  // __________________________________________________________________
  // recurrent layers
  class EmbeddingLayer : public IRecurrentLayer
  {
  public:
    EmbeddingLayer(int var_row_index, MatrixXd W);
    virtual ~EmbeddingLayer() {};
    virtual MatrixXd scan( const MatrixXd&);

  private:
    int _var_row_index;
    MatrixXd _W;
  };


  /// long short term memory ///
  class LSTMLayer : public IRecurrentLayer
  {
  public:
    LSTMLayer(Activation activation, Activation inner_activation,
        MatrixXd W_i, MatrixXd U_i, VectorXd b_i,
        MatrixXd W_f, MatrixXd U_f, VectorXd b_f,
        MatrixXd W_o, MatrixXd U_o, VectorXd b_o,
        MatrixXd W_c, MatrixXd U_c, VectorXd b_c,
        bool return_sequences = true);

    virtual ~LSTMLayer() {};
    virtual VectorXd step( const VectorXd&);
    virtual MatrixXd scan( const MatrixXd&);

  private:
    std::function<double(double)> _activation_fun;
    std::function<double(double)> _inner_activation_fun;

    MatrixXd _W_i;
    MatrixXd _U_i;
    VectorXd _b_i;

    MatrixXd _W_f;
    MatrixXd _U_f;
    VectorXd _b_f;

    MatrixXd _W_o;
    MatrixXd _U_o;
    VectorXd _b_o;

    MatrixXd _W_c;
    MatrixXd _U_c;
    VectorXd _b_c;

    //states
    MatrixXd _C_t;
    MatrixXd _h_t;
    int _time;

    int _n_outputs;

    bool _return_sequences;
  };

  class RecurrentStack
  {
  public:
    RecurrentStack(size_t n_inputs, const std::vector<LayerConfig>& layers);
    ~RecurrentStack();
    RecurrentStack(RecurrentStack&) = delete;
    RecurrentStack& operator=(RecurrentStack&) = delete;
    VectorXd reduce(MatrixXd inputs) const;
    size_t n_outputs() const;
  private:
    std::vector<IRecurrentLayer*> _layers;
    size_t add_lstm_layers(size_t n_inputs, const LayerConfig&);
    size_t add_embedding_layers(size_t n_inputs, const LayerConfig&);
    Stack* _stack;
  };


}





#endif
