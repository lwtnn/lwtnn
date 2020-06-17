#include "lwtnn/LightweightNeuralNetwork.hh"
#include "lwtnn/InputPreprocessor.hh"
#include "lwtnn/Stack.hh"
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
  template<typename T>
  LightweightNeuralNetworkT<T>::LightweightNeuralNetworkT(
    const std::vector<Input>& inputs,
    const std::vector<LayerConfig>& layers,
    const std::vector<std::string>& outputs):
    m_stack(new StackT<T>(inputs.size(), layers)),
    m_preproc(new InputPreprocessorT<T>(inputs)),
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
  LightweightNeuralNetworkT<T>::~LightweightNeuralNetworkT() {
    delete m_stack;
    m_stack = 0;
    delete m_preproc;
    m_preproc = 0;
  }

  template<typename T>
  lwt::ValueMap
  LightweightNeuralNetworkT<T>::compute(const lwt::ValueMap& in) const {

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
  LightweightRNNT<T>::LightweightRNNT(const std::vector<Input>& inputs,
                                 const std::vector<LayerConfig>& layers,
                                 const std::vector<std::string>& outputs):
    m_stack(new ReductionStack(inputs.size(), layers)),
    m_preproc(new InputPreprocessor(inputs)),
    m_vec_preproc(new InputVectorPreprocessor(inputs)),
    m_outputs(outputs.begin(), outputs.end()),
    m_n_inputs(inputs.size())
  {
    if (m_outputs.size() != m_stack->n_outputs()) {
      throw NNConfigurationException(
        "Mismatch between NN output dimensions and output labels");
    }
  }
  
  template<typename T>
  LightweightRNNT<T>::~LightweightRNNT() {
    delete m_stack;
    delete m_preproc;
    delete m_vec_preproc;
  }

  template<typename T>
  ValueMap LightweightRNNT<T>::reduce(const std::vector<ValueMap>& in) const {
    const auto& preproc = *m_preproc;
    MatrixXd inputs(m_n_inputs, in.size());
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
  ValueMap LightweightRNNT<T>::reduce(const VectorMap& in) const {
    const auto& preproc = *m_vec_preproc;
    auto outvec = m_stack->reduce(preproc(in));
    ValueMap out;
    const auto n_rows = static_cast<size_t>(outvec.rows());
    for (size_t iii = 0; iii < n_rows; iii++) {
      out.emplace(m_outputs.at(iii), outvec(iii));
    }
    return out;
  }


}
