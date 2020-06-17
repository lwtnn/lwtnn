#include "lwtnn/InputPreprocessor.hh"

namespace lwt {
  // ______________________________________________________________________
  // Input preprocessors

  // simple feed-forwared version
  
  template<typename T>
  InputPreprocessorT<T>::InputPreprocessorT(const std::vector<Input>& inputs):
    m_offsets(inputs.size()),
    m_scales(inputs.size())
  {
    static_assert( std::is_same<T, double>::value ||
                   std::is_assignable<T, double>::value, 
                   "double cannot be implicitly assigned to T" );
    
    size_t in_num = 0;
    for (const auto& input: inputs) {
      m_offsets(in_num) = input.offset;
      m_scales(in_num) = input.scale;
      m_names.push_back(input.name);
      in_num++;
    }
  }
  
  template<typename T>
  VectorX<T> InputPreprocessorT<T>::operator()(const ValueMap& in) const {
    VectorX<T> invec(m_names.size());
    size_t input_number = 0;
    for (const auto& in_name: m_names) {
      if (!in.count(in_name)) {
        throw NNEvaluationException("can't find input: " + in_name);
      }
      invec(input_number) = static_cast<T>(in.at(in_name));
      input_number++;
    }
    return (invec + m_offsets).cwiseProduct(m_scales);
  }


  // Input vector preprocessor
  template<typename T>
  InputVectorPreprocessorT<T>::InputVectorPreprocessorT(
    const std::vector<Input>& inputs):
    m_offsets(inputs.size()),
    m_scales(inputs.size())
  {
    static_assert( std::is_same<T, double>::value ||
                   std::is_assignable<T, double>::value, 
                   "double cannot be implicitly assigned to T" );
    
    size_t in_num = 0;
    for (const auto& input: inputs) {
      m_offsets(in_num) = input.offset;
      m_scales(in_num) = input.scale;
      m_names.push_back(input.name);
      in_num++;
    }
    // require at least one input at configuration, since we require
    // at least one for evaluation
    if (in_num == 0) {
      throw NNConfigurationException("need at least one input");
    }
  }
  
  template<typename T>
  MatrixX<T> InputVectorPreprocessorT<T>::operator()(const VectorMap& in) const {
    using namespace Eigen;
    if (in.size() == 0) {
      throw NNEvaluationException("Empty input map");
    }
    size_t n_cols = in.begin()->second.size();
    MatrixX<T> inmat(m_names.size(), n_cols);
    size_t in_num = 0;
    for (const auto& in_name: m_names) {
      if (!in.count(in_name)) {
        throw NNEvaluationException("can't find input: " + in_name);
      }
      const auto& invec = in.at(in_name);
      if (invec.size() != n_cols) {
        throw NNEvaluationException("Input vector size mismatch");
      }
      inmat.row(in_num) = Map<const VectorXd>(invec.data(), invec.size());
      in_num++;
    }
    if (n_cols == 0) {
        return MatrixXd(m_names.size(), 0);
    }
    return m_scales.asDiagonal() * (inmat.colwise() + m_offsets);
  }

}
