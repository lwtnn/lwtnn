#ifndef LWTNN_STACK_HH
#define LWTNN_STACK_HH

#include "lwtnn/generic/Stack.hh"

namespace lwt
{
  // double typedef for feed-forward stack
  using Stack = generic::Stack<double>;

  // double typedefs for feed-forward layers
  using ILayer = generic::ILayer<double>;
  using DummyLayer = generic::DummyLayer<double>;
  using UnaryActivationLayer = generic::UnaryActivationLayer<double>;
  using SoftmaxLayer = generic::SoftmaxLayer<double>;
  using BiasLayer = generic::BiasLayer<double>;
  using MatrixLayer = generic::MatrixLayer<double>;
  using MaxoutLayer = generic::MaxoutLayer<double>;
  using NormalizationLayer = generic::NormalizationLayer<double>;
  using HighwayLayer = generic::HighwayLayer<double>;

  // double typedefs for recurrent stack
  using RecurrentStack = generic::RecurrentStack<double>;
  using ReductionStack = generic::ReductionStack<double>;

  // double typedefs for recurrent layesr
  using IRecurrentLayer = generic::IRecurrentLayer<double>;
  using EmbeddingLayer = generic::EmbeddingLayer<double>;
  using LSTMLayer = generic::LSTMLayer<double>;
  using GRULayer = generic::GRULayer<double>;
  using SimpleRNNLayer = generic::SimpleRNNLayer<double>;
  using DenseComponents = generic::DenseComponents<double>;

  // double typedefs for activation functions
  using ELU = generic::ELU<double>;
  using LeakyReLU = generic::LeakyReLU<double>;
  using Swish = generic::Swish<double>;

  double nn_sigmoid( double x );
  double nn_hard_sigmoid( double x );
  double nn_tanh( double x );
  double nn_relu( double x );

  // double typedefs for utility functions
  MatrixX<double> build_matrix(const std::vector<double>& weights, std::size_t n_inputs);
  VectorX<double> build_vector(const std::vector<double>& bias);

  DenseComponents get_component(const lwt::LayerConfig& layer, std::size_t n_in);
  std::function<double(double)> get_activation(lwt::ActivationConfig);

}

#endif
