#ifndef LIGHTWEIGHT_NEURAL_NETWORK_HH
#define LIGHTWEIGHT_NEURAL_NETWORK_HH

// Lightweight Neural Networks
//
// This is a simple NN implementation, designed to be lightweight in
// terms of both size and dependencies. For sanity we use Eigen, but
// otherwise this aims to be a minimal NN class which is fully
// configurable at runtime.
//
// The classes defined here are the high level wrappers: they don't
// directly include any Eigen code (to speed compliation of algorithms
// that use them), and they store data in STL objects.
//
// Authors: Dan Guest <dguest@cern.ch>
//          Michael Kagan <mkagan@cern.ch>
//          Michela Paganini <micky.91@hotmail.com>

#include <memory>
#include "lightweight_network_config.hh"

namespace lwt {

  // use a normal map externally, since these are more common in user
  // code.
  // TODO: is it worth changing to unordered_map?
  typedef std::map<std::string, double> ValueMap;
  typedef std::vector<std::pair<std::string, double> > ValueVector;
  typedef std::map<std::string, std::vector<double> > VectorMap;
  
  // ______________________________________________________________________
  // high-level wrappers

  namespace generic {
    template<typename T> class LightweightNeuralNetwork;
    template<typename T> class LightweightRNN;
  }
  
  // feed-forward variant
  class LightweightNeuralNetwork
  {
  public:
    LightweightNeuralNetwork(const std::vector<Input>& inputs,
                             const std::vector<LayerConfig>& layers,
                             const std::vector<std::string>& outputs);
    
    ~LightweightNeuralNetwork();
    
    // disable copying until we need it...
    LightweightNeuralNetwork(LightweightNeuralNetwork&) = delete;
    LightweightNeuralNetwork& operator=(LightweightNeuralNetwork&) = delete;

    // use a normal map externally, since these are more common in
    // user code.
    // TODO: is it worth changing to unordered_map?
    ValueMap compute(const ValueMap&) const;

  private:
    std::unique_ptr<generic::LightweightNeuralNetwork<double>> m_impl;
  };
  
  // recurrent version
  class LightweightRNN
  {
  public:
    LightweightRNN(const std::vector<Input>& inputs,
                   const std::vector<LayerConfig>& layers,
                   const std::vector<std::string>& outputs);
    
    ~LightweightRNN();
    
    LightweightRNN(LightweightRNN&) = delete;
    LightweightRNN& operator=(LightweightRNN&) = delete;

    ValueMap reduce(const std::vector<ValueMap>&) const;
    ValueMap reduce(const VectorMap&) const;
  private:
    std::unique_ptr<generic::LightweightRNN<double>> m_impl;
  };


}
#endif
