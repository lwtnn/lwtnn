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

#include "NNLayerConfig.hh"

namespace lwt {

  class Stack;
  class RecurrentStack;
  class InputPreprocessor;
  class InputVectorPreprocessor;

  // use a normal map externally, since these are more common in user
  // code.
  // TODO: is it worth changing to unordered_map?
  typedef std::map<std::string, double> ValueMap;
  typedef std::vector<std::pair<std::string, double> > ValueVector;
  typedef std::map<std::string, std::vector<double> > VectorMap;

  // ______________________________________________________________________
  // high-level wrappers

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
    // use the Stack class above as the computational core
    Stack* m_stack;
    InputPreprocessor* m_preproc;

    // output labels
    std::vector<std::string> m_outputs;

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
    RecurrentStack* m_stack;
    InputPreprocessor* m_preproc;
    InputVectorPreprocessor* m_vec_preproc;
    std::vector<std::string> m_outputs;
    size_t m_n_inputs;
  };

}
#endif
