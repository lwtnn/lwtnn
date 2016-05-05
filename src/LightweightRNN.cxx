// Lightweight Recurrent NN
//
//basic code for forward pass computation of recurrent NN structures,
// like LSTM, useful for processing time series / sequence data.  goal
// to be able to evaluate Keras (keras.io) models in c++ in
// lightweight way
//
// Author: Michael Kagan <mkagan@cern.ch>


#include "lwtnn/LightweightRNN.hh"


namespace {
  // LSTM component for convenience
  // TODO: consider using this in LSTMLayer
  struct LSTMComponent
  {
    Eigen::MatrixXd W;
    Eigen::MatrixXd U;
    Eigen::VectorXd b;
  };
  LSTMComponent get_component(const lwt::LayerConfig& layer, size_t n_in);

  std::function<double(double)> get_activation(lwt::Activation);
}


namespace lwt {


  EmbeddingLayer::EmbeddingLayer(int var_row_index, MatrixXd W):
    _var_row_index(var_row_index),
    _W(W)
  {
    if(var_row_index < 0)
      throw NNConfigurationException(
        "EmbeddingLayer::EmbeddingLayer - can not set var_row_index<0,"
        " it is an index for a matrix row!");
  }



  MatrixXd EmbeddingLayer::scan( const MatrixXd& x) {

    if( _var_row_index >= x.rows() )
      throw NNEvaluationException(
        "EmbeddingLayer::scan - var_row_index is larger than input matrix"
        " number of rows!");

    MatrixXd embedded(_W.rows(), x.cols());

    for(int icol=0; icol<x.cols(); icol++)
      embedded.col(icol) = _W.col( x(_var_row_index, icol) );

    //only embed 1 variable at a time, so this should be correct size
    MatrixXd out(_W.rows() + (x.rows() - 1), x.cols());

    //assuming _var_row_index is an index with first possible value of 0
    if(_var_row_index > 0)
      out.topRows(_var_row_index) = x.topRows(_var_row_index);

    out.block(_var_row_index, 0, embedded.rows(), embedded.cols()) = embedded;

    if( _var_row_index < (x.rows()-1) )
      out.bottomRows( x.cols() - 1 - _var_row_index)
        = x.bottomRows( x.cols() - 1 - _var_row_index);

    return out;
  }

  LSTMLayer::LSTMLayer(Activation activation, Activation inner_activation,
           MatrixXd W_i, MatrixXd U_i, VectorXd b_i,
           MatrixXd W_f, MatrixXd U_f, VectorXd b_f,
           MatrixXd W_o, MatrixXd U_o, VectorXd b_o,
           MatrixXd W_c, MatrixXd U_c, VectorXd b_c,
           bool return_sequences):
    _W_i(W_i),
    _U_i(U_i),
    _b_i(b_i),
    _W_f(W_f),
    _U_f(U_f),
    _b_f(b_f),
    _W_o(W_o),
    _U_o(U_o),
    _b_o(b_o),
    _W_c(W_c),
    _U_c(U_c),
    _b_c(b_c),
    _return_sequences(return_sequences)
  {
    _n_outputs = _W_o.rows();

    _activation_fun = get_activation(activation);
    _inner_activation_fun = get_activation(inner_activation);
  }

  VectorXd LSTMLayer::step( const VectorXd& x_t ) {
    // https://github.com/fchollet/keras/blob/master/keras/layers/recurrent.py#L740

    if(_time < 0)
      throw NNEvaluationException(
        "LSTMLayer::compute - time is less than zero!");

    const auto& act_fun = _activation_fun;
    const auto& in_act_fun = _inner_activation_fun;

    int tm1 = std::max(0, _time - 1);
    VectorXd h_tm1 = _h_t.col(tm1);
    VectorXd C_tm1 = _C_t.col(tm1);

    VectorXd i  =  (_W_i*x_t + _b_i + _U_i*h_tm1).unaryExpr(in_act_fun);
    VectorXd f  =  (_W_f*x_t + _b_f + _U_f*h_tm1).unaryExpr(in_act_fun);
    VectorXd o  =  (_W_o*x_t + _b_o + _U_o*h_tm1).unaryExpr(in_act_fun);
    VectorXd ct =  (_W_c*x_t + _b_c + _U_c*h_tm1).unaryExpr(act_fun);

    _C_t.col(_time) = f.cwiseProduct(C_tm1) + i.cwiseProduct(ct);
    _h_t.col(_time) = o.cwiseProduct( _C_t.col(_time).unaryExpr(act_fun) );

    return VectorXd( _h_t.col(_time) );
  }

  MatrixXd LSTMLayer::scan( const MatrixXd& x ){

    _C_t.resize(_n_outputs, x.cols());
    _C_t.setZero();
    _h_t.resize(_n_outputs, x.cols());
    _h_t.setZero();
    _time = -1;


    for(_time=0; _time < x.cols(); _time++)
      {
  this->step( x.col( _time ) );
      }

    return _return_sequences ? _h_t : _h_t.col(_h_t.cols() - 1);
  }

  // ______________________________________________________________________
  // Recurrent Stack

  RecurrentStack::RecurrentStack(size_t n_inputs,
                                 const std::vector<lwt::LayerConfig>& layers)
  {
    using namespace lwt;
    size_t layer_n = 0;
    const size_t n_layers = layers.size();
    for (;layer_n < n_layers; layer_n++) {
      auto& layer = layers.at(layer_n);

      // add recurrent layers (now only LSTM)
      if (layer.architecture == Architecture::LSTM) {
        n_inputs = add_lstm_layers(n_inputs, layer);
      } else if (layer.architecture == Architecture::EMBEDDING) {
        n_inputs = add_embedding_layers(n_inputs, layer);
      } else {
        // leave this loop if we're done with the recurrent stuff
        break;
      }
    }
    // fill the remaining dense layers
    _stack = new Stack(n_inputs, layers, layer_n);
  }
  RecurrentStack::~RecurrentStack() {
    for (auto& layer: _layers) {
      delete layer;
      layer = 0;
    }
    delete _stack;
    _stack = 0;
  }
  VectorXd RecurrentStack::reduce(MatrixXd in) const {
    for (auto* layer: _layers) {
      in = layer->scan(in);
    }
    return _stack->compute(in.col(in.cols() - 1));
  }
  size_t RecurrentStack::n_outputs() const {
    return _stack->n_outputs();
  }

  size_t RecurrentStack::add_lstm_layers(size_t n_inputs,
                                         const LayerConfig& layer) {
    auto& comps = layer.components;
    const auto& i = get_component(comps.at(Component::I), n_inputs);
    const auto& o = get_component(comps.at(Component::O), n_inputs);
    const auto& f = get_component(comps.at(Component::F), n_inputs);
    const auto& c = get_component(comps.at(Component::C), n_inputs);
    _layers.push_back(
      new LSTMLayer(layer.activation, layer.inner_activation,
                    i.W, i.U, i.b,
                    f.W, f.U, f.b,
                    o.W, o.U, o.b,
                    c.W, c.U, c.b));
    return o.b.rows();
  }

  size_t RecurrentStack::add_embedding_layers(size_t n_inputs,
                                              const LayerConfig& layer) {
    for (const auto& emb: layer.embedding) {
      size_t n_wt = emb.weights.size();
      size_t n_cats = n_wt / emb.n_out;
      MatrixXd mat = build_matrix(emb.weights, n_cats);
      _layers.push_back(new EmbeddingLayer(emb.index, mat));
      n_inputs += emb.n_out - 1;
    }
    return n_inputs;
  }

  // ______________________________________________________________________
  // Activation functions
  double nn_sigmoid( double x ){
    //https://github.com/Theano/Theano/blob/master/theano/tensor/nnet/sigm.py#L35

    if( x< -30.0 )
      return 0.0;

    if( x > 30.0 )
      return 1.0;

    return 1.0 / (1.0 + std::exp(-1.0*x));

  }

  double nn_hard_sigmoid( double x ){
    //https://github.com/Theano/Theano/blob/master/theano/tensor/nnet/sigm.py#L279

    double out = 0.2*x + 0.5;

    if( out < 0)
      return 0.0;

    if ( out > 1 )
      return 1.0;

    return out;
  }

  double nn_tanh( double x ){
    return std::tanh(x);
  }

}

// ________________________________________________________________________
// convenience functions

namespace {
  LSTMComponent get_component(const lwt::LayerConfig& layer, size_t n_in) {
    using namespace Eigen;
    using namespace lwt;
    MatrixXd weights = build_matrix(layer.weights, n_in);
    size_t n_out = weights.rows();
    MatrixXd U = build_matrix(layer.U, n_out);
    VectorXd bias = build_vector(layer.bias);

    size_t u_out = U.rows();
    size_t b_out = bias.rows();
    if (u_out != n_out || b_out != n_out) {
      throw NNConfigurationException(
        "Output dims mismatch, W: " + std::to_string(n_out) +
        ", U: " + std::to_string(u_out) + ", b: " + std::to_string(b_out));
    }
    return {weights, U, bias};
  }
  std::function<double(double)> get_activation(lwt::Activation act) {
    using namespace lwt;
    switch (act) {
    case Activation::SIGMOID: return nn_sigmoid;
    case Activation::HARD_SIGMOID: return nn_hard_sigmoid;
    case Activation::TANH: return nn_tanh;
    default: {
      throw NNConfigurationException("Got undefined activation function");
    }
    }
  }

}
