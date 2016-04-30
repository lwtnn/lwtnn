// Lightweight Recurrent NN
//
//basic code for forward pass computation of recurrent NN structures, like LSTM,
// useful for processing time series / sequence data.
// goal to be able to evaluate Keras (keras.io) models in c++ in lightweight way
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
}


namespace lwt {


  MatrixXd MaskingLayer::scan( const MatrixXd& x) {
    set_mask(   (x.colwise().sum().array() == 0).cast<int>()   );
    return x;
  }

  MatrixXd EmbeddingLayer::scan( const MatrixXd& x) {
    MatrixXd out(_W.rows(), x.cols());

    for(int icol=0; icol<x.cols(); icol++)
      out.col(icol) = _W.col( x(0, icol) ) + _b;

    return out;
  }

  MatrixXd TimeDistributedMergeLayer::scan(const MatrixXd& X1, const MatrixXd& X2) {

    if(X1.cols() != X2.cols())
      throw NNEvaluationException("TimeDistributedMergeLayer::scan - Matrices do not have same number of columns (time-dim.)");

    MatrixXd out(X1.rows()+X2.rows(), X1.cols());
    out << X1,
           X2;
    // need to check that this concatenates properly...

    return out;
  }

  LSTMLayer::LSTMLayer(Activation activation, Activation inner_activation,
           MatrixXd W_i, MatrixXd U_i, VectorXd b_i,
           MatrixXd W_f, MatrixXd U_f, VectorXd b_f,
           MatrixXd W_o, MatrixXd U_o, VectorXd b_o,
           MatrixXd W_c, MatrixXd U_c, VectorXd b_c ):
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
    _b_c(b_c)
  {
    _n_inputs  = _W_o.cols();
    _n_outputs = _W_o.rows();

    if(activation==Activation::SIGMOID)           _activation_fun = nn_sigmoid;
    else if(activation==Activation::HARD_SIGMOID) _activation_fun = nn_hard_sigmoid;
    else if(activation==Activation::TANH)         _activation_fun = nn_tanh;

    if(inner_activation==Activation::SIGMOID)           _inner_activation_fun = nn_sigmoid;
    else if(inner_activation==Activation::HARD_SIGMOID) _inner_activation_fun = nn_hard_sigmoid;
    else if(inner_activation==Activation::TANH)         _inner_activation_fun = nn_tanh;

  }

  VectorXd LSTMLayer::step( const VectorXd& x_t ) {
    // https://github.com/fchollet/keras/blob/master/keras/layers/recurrent.py#L740

    if(_time < 0)
      throw NNEvaluationException("LSTMLayer::compute - time is less than zero!");

    if(_time == 0){
      if( get_mask()(_time) == 1 ){
  //_C_t.col(_time).setZero();
  //_h_t.col(_time).setZero();
  return VectorXd( _h_t.col(_time) );
      }

      VectorXd i =  (_W_i*x_t + _b_i).unaryExpr(_inner_activation_fun);
      VectorXd f =  (_W_f*x_t + _b_f).unaryExpr(_inner_activation_fun);
      VectorXd o =  (_W_o*x_t + _b_o).unaryExpr(_inner_activation_fun);
      _C_t.col(_time) = i.cwiseProduct(  (_W_c*x_t + _b_c).unaryExpr(_activation_fun)  );
      _h_t.col(_time) = o.cwiseProduct( _C_t.col(_time).unaryExpr(_activation_fun) );
    }

    else{
      if( get_mask()(_time) == 1 ){
  _C_t.col(_time) = _C_t.col(_time - 1);
  _h_t.col(_time) = _h_t.col(_time - 1);
  return VectorXd( _h_t.col(_time) );
      }

      VectorXd h_tm1 = _h_t.col(_time - 1);
      VectorXd C_tm1 = _C_t.col(_time - 1);

      VectorXd i =  (_W_i*x_t + _b_i + _U_i*h_tm1).unaryExpr(_inner_activation_fun);
      VectorXd f =  (_W_f*x_t + _b_f + _U_f*h_tm1).unaryExpr(_inner_activation_fun);
      VectorXd o =  (_W_o*x_t + _b_o + _U_o*h_tm1).unaryExpr(_inner_activation_fun);
      _C_t.col(_time) = f.cwiseProduct(C_tm1) + i.cwiseProduct(  (_W_c*x_t + _b_c + _U_c*h_tm1).unaryExpr(_activation_fun)  );
      _h_t.col(_time) = o.cwiseProduct( _C_t.col(_time).unaryExpr(_activation_fun) );
    }

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

    return MatrixXd(_h_t);
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
      n_inputs = add_lstm_layers(n_inputs, layer);
      // TODO: add break if this becomes a dense layer
    }
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
}
