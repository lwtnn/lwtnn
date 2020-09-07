#ifndef LWTNN_GENERIC_LIGHTWEIGHT_NEURAL_NETWORK_TCC
#define LWTNN_GENERIC_LIGHTWEIGHT_NEURAL_NETWORK_TCC

#include "lwtnn/generic/LightweightNeuralNetwork.hh"
#include "lwtnn/generic/InputPreprocessor.hh"
#include "lwtnn/generic/Stack.hh"
#include "lwtnn/generic/eigen_typedefs.hh"

#include <set>

namespace lwt {
namespace generic {

  // ______________________________________________________________________
  // LightweightNeuralNetwork HL wrapper
  template<typename T>
  LightweightNeuralNetwork<T>::LightweightNeuralNetwork(
    const std::vector<Input>& inputs,
    const std::vector<LayerConfig>& layers,
    const std::vector<std::string>& outputs):
    m_stack(new Stack<T>(inputs.size(), layers)),
    m_preproc(new InputPreprocessor<T>(inputs)),
    m_outputs(outputs.begin(), outputs.end())
  {
    if (m_outputs.size() != m_stack->n_outputs()) {
      std::string problem = "internal stack has " +
        std::to_string(m_stack->n_outputs()) + " outputs, but " +
        std::to_string(m_outputs.size()) + " were given";
      throw NNConfigurationException(problem);
    }
  }

  template<typename T>
  LightweightNeuralNetwork<T>::~LightweightNeuralNetwork() {
    delete m_stack;
    m_stack = 0;
    delete m_preproc;
    m_preproc = 0;
  }

  template<typename T>
  lwt::ValueMap
  LightweightNeuralNetwork<T>::compute(const lwt::ValueMap& in) const {

    // compute outputs
    const auto& preproc = *m_preproc;
    auto outvec = m_stack->compute(preproc(in));
    assert(outvec.rows() > 0);
    auto out_size = static_cast<size_t>(outvec.rows());
    assert(out_size == m_outputs.size());

    // build and return output map
    lwt::ValueMap out_map;
    for (size_t out_n = 0; out_n < out_size; out_n++) {
      out_map.emplace(m_outputs.at(out_n), outvec(out_n));
    }
    return out_map;
  }

  // ______________________________________________________________________
  // LightweightRNN

  template<typename T>
  LightweightRNN<T>::LightweightRNN(const std::vector<Input>& inputs,
                                 const std::vector<LayerConfig>& layers,
                                 const std::vector<std::string>& outputs):
    m_stack(new generic::ReductionStack<T>(inputs.size(), layers)),
    m_preproc(new generic::InputPreprocessor<T>(inputs)),
    m_vec_preproc(new generic::InputVectorPreprocessor<T>(inputs)),
    m_outputs(outputs.begin(), outputs.end()),
    m_n_inputs(inputs.size())
  {
    if (m_outputs.size() != m_stack->n_outputs()) {
      throw NNConfigurationException(
        "Mismatch between NN output dimensions and output labels");
    }
  }
  
  template<typename T>
  LightweightRNN<T>::~LightweightRNN() {
    delete m_stack;
    delete m_preproc;
    delete m_vec_preproc;
  }

  template<typename T>
  ValueMap LightweightRNN<T>::reduce(const std::vector<ValueMap>& in) const {
    const auto& preproc = *m_preproc;
    MatrixX<T> inputs(m_n_inputs, in.size());
    for (size_t iii = 0; iii < in.size(); iii++) {
      inputs.col(iii) = preproc(in.at(iii));
    }
    auto outvec = m_stack->reduce(inputs);
    ValueMap out;
    const auto n_rows = static_cast<size_t>(outvec.rows());
    for (size_t iii = 0; iii < n_rows; iii++) {
      out.emplace(m_outputs.at(iii), outvec(iii));
    }
    return out;
  }

  // this version should be slightly faster since it only has to sort
  // the inputs once
  
  template<typename T>
  ValueMap LightweightRNN<T>::reduce(const VectorMap& in) const {
    const auto& preproc = *m_vec_preproc;
    auto outvec = m_stack->reduce(preproc(in));
    ValueMap out;
    const auto n_rows = static_cast<size_t>(outvec.rows());
    for (size_t iii = 0; iii < n_rows; iii++) {
      out.emplace(m_outputs.at(iii), outvec(iii));
    }
    return out;
  }
} // namespace generic
} // namespace lwt

#endif
