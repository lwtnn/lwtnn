#ifndef LWTNN_GENERIC_INPUT_PREPROCESSOR_HH
#define LWTNN_GENERIC_INPUT_PREPROCESSOR_HH

#include "lwtnn/lightweight_network_config.hh"
#include "lwtnn/Exceptions.hh"
#include "lwtnn/generic/eigen_typedefs.hh"

#include <map>
#include <vector>

namespace lwt {

  // use a normal map externally, since these are more common in user
  // code.  TODO: is it worth changing to unordered_map?
  typedef std::map<std::string, double> ValueMap;
  typedef std::vector<std::pair<std::string, double> > ValueVector;
  typedef std::map<std::string, std::vector<double> > VectorMap;

namespace generic {

  // ______________________________________________________________________
  // input preprocessor (handles normalization and packing into Eigen)

  template<typename T>
  class InputPreprocessor
  {
  public:
    InputPreprocessor(const std::vector<Input>& inputs);
    VectorX<T> operator()(const ValueMap&) const;
  private:
    // input transformations
    VectorX<T> m_offsets;
    VectorX<T> m_scales;
    std::vector<std::string> m_names;
  };

  template<typename T>
  class InputVectorPreprocessor
  {
  public:
    InputVectorPreprocessor(const std::vector<Input>& inputs);
    MatrixX<T> operator()(const VectorMap&) const;
  private:
    // input transformations
    VectorX<T> m_offsets;
    VectorX<T> m_scales;
    std::vector<std::string> m_names;
  };

} // namespace generic

} // namespace lwt

#include "InputPreprocessor.tcc"

#endif // LWTNN_GENERIC_INPUT_PREPROCESSOR_HH
