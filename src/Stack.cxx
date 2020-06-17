#include "lwtnn/Stack.hh"
#include <Eigen/Dense>

namespace lwt
{
  // Definitions for double overloads
    
  ILayer* get_raw_activation_layer(ActivationConfig activation) {
    return get_raw_activation_layerT<double>(activation);
  }
  
  std::function<double(double)> get_activation(lwt::ActivationConfig act) {
    return get_activationT<double>(act);
  }
  
  double nn_sigmoid( double x ) {
    return nn_sigmoidT<double>(x);
  }

  double nn_hard_sigmoid( double x ) {
    return nn_hard_sigmoidT<double>(x);
  }
  
  double nn_tanh( double x ) {
    return nn_tanhT<double>(x);
  }

  double nn_relu( double x) {
    return nn_reluT<double>(x);
  }
    
  MatrixX<double> build_matrix(const std::vector<double>& weights, size_t n_inputs) {
    return build_matrixT<double, double>(weights, n_inputs);
  }
  
  VectorX<double> build_vector(const std::vector<double>& bias) {
    return build_vectorT<double, double>(bias);
  }
  
  DenseComponents get_component(const lwt::LayerConfig& layer, size_t n_in) {
    return get_componentT<double>(layer, n_in);
  }

  // consistency checks
  void throw_if_not_maxout(const LayerConfig& layer) {
    bool wt_ok = layer.weights.size() == 0;
    bool bias_ok = layer.bias.size() == 0;
    bool maxout_ok = layer.sublayers.size() > 0;
    bool act_ok = layer.activation.function == Activation::NONE;
    if (wt_ok && bias_ok && maxout_ok && act_ok) return;
    throw NNConfigurationException("layer has wrong info for maxout");
  }
  void throw_if_not_dense(const LayerConfig& layer) {
    if (layer.sublayers.size() > 0) {
      throw NNConfigurationException("sublayers in dense layer");
    }
  }

  void throw_if_not_normalization(const LayerConfig& layer) {
    if (layer.sublayers.size() > 0) {
      throw NNConfigurationException("sublayers in normalization layer");
    }
  }
}
