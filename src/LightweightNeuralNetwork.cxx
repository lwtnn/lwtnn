#include "lwtnn/LightweightNeuralNetwork.hh"
#include "lwtnn/generic/LightweightNeuralNetwork.hh"
#include "lwtnn/generic/InputPreprocessor.hh"
#include "lwtnn/generic/Stack.hh"

#include <Eigen/Dense>

#include <set>

// internal utility functions
namespace {
  using namespace Eigen;
  using namespace lwt;
}
namespace lwt {

  // ______________________________________________________________________
  // LightweightNeuralNetwork HL wrapper
  LightweightNeuralNetwork::LightweightNeuralNetwork(
    const std::vector<Input>& inputs,
    const std::vector<LayerConfig>& layers,
    const std::vector<std::string>& outputs):
    m_impl(new generic::LightweightNeuralNetwork<double>(inputs, layers, outputs))
  {
  }

  LightweightNeuralNetwork::~LightweightNeuralNetwork()
  {
  }

  lwt::ValueMap
  LightweightNeuralNetwork::compute(const lwt::ValueMap& in) const {
    return m_impl->compute(in);
  }

  // ______________________________________________________________________
  // LightweightRNN

  LightweightRNN::LightweightRNN(const std::vector<Input>& inputs,
                                 const std::vector<LayerConfig>& layers,
                                 const std::vector<std::string>& outputs):
    m_impl(new generic::LightweightRNN<double>(inputs, layers, outputs))
  {
  }

  LightweightRNN::~LightweightRNN()
  {
  }

  ValueMap LightweightRNN::reduce(const std::vector<ValueMap>& in) const {
    return m_impl->reduce(in);
  }

  // this version should be slightly faster since it only has to sort
  // the inputs once
  ValueMap LightweightRNN::reduce(const VectorMap& in) const {
    return m_impl->reduce(in);
  }
}
