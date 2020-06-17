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

#include "lightweight_network_config.hh"

namespace lwt {

  template<typename T> class StackT;
  template<typename T> class ReductionStackT;
  
  using Stack = StackT<double>;
  using ReductionStack = ReductionStackT<double>;
  
  template<typename T> class InputPreprocessorT;
  template<typename T> class InputVectorPreprocessorT;
  
  using InputPreprocessor = InputPreprocessorT<double>;
  using InputVectorPreprocessor = InputVectorPreprocessorT<double>;

  // use a normal map externally, since these are more common in user
  // code.
  // TODO: is it worth changing to unordered_map?
  typedef std::map<std::string, double> ValueMap;
  typedef std::vector<std::pair<std::string, double> > ValueVector;
  typedef std::map<std::string, std::vector<double> > VectorMap;

  // ______________________________________________________________________
  // high-level wrappers

  // feed-forward variant
  template<typename T>
  class LightweightNeuralNetworkT
  {
  public:
    LightweightNeuralNetworkT(const std::vector<Input>& inputs,
                             const std::vector<LayerConfig>& layers,
                             const std::vector<std::string>& outputs);
    ~LightweightNeuralNetworkT();
    // disable copying until we need it...
    LightweightNeuralNetworkT(LightweightNeuralNetworkT&) = delete;
    LightweightNeuralNetworkT& operator=(LightweightNeuralNetworkT&) = delete;

    // use a normal map externally, since these are more common in
    // user code.
    // TODO: is it worth changing to unordered_map?
    ValueMap compute(const ValueMap&) const;

  protected:
    // use the Stack class above as the computational core
    StackT<T>* m_stack;
    InputPreprocessorT<T>* m_preproc;

    // output labels
    std::vector<std::string> m_outputs;
  };
  
  using LightweightNeuralNetwork = LightweightNeuralNetworkT<double>;

  // recurrent version
  template<typename T>
  class LightweightRNNT
  {
  public:
    LightweightRNNT(const std::vector<Input>& inputs,
                   const std::vector<LayerConfig>& layers,
                   const std::vector<std::string>& outputs);
    ~LightweightRNNT();
    LightweightRNNT(LightweightRNNT&) = delete;
    LightweightRNNT& operator=(LightweightRNNT&) = delete;

    ValueMap reduce(const std::vector<ValueMap>&) const;
    ValueMap reduce(const VectorMap&) const;
  private:
    ReductionStackT<T>* m_stack;
    InputPreprocessorT<T>* m_preproc;
    InputVectorPreprocessorT<T>* m_vec_preproc;
    std::vector<std::string> m_outputs;
    size_t m_n_inputs;
  };

  using LightweightRNN = LightweightRNNT<double>;
}

#include "LightweightNeuralNetwork.txx"

#endif
