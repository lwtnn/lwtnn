#ifndef INPUT_PREPROCESSOR_HH
#define INPUT_PREPROCESSOR_HH

#include "lwtnn/lightweight_network_config.hh"
#include "lwtnn/Exceptions.hh"

#include <Eigen/Dense>
#include <map>
#include <vector>

namespace lwt {

  template<typename T>
  using VectorX = Eigen::Matrix<T, Eigen::Dynamic, 1>;
  
  template<typename T>
  using MatrixX = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

  // use a normal map externally, since these are more common in user
  // code.  TODO: is it worth changing to unordered_map?
  typedef std::map<std::string, double> ValueMap;
  typedef std::vector<std::pair<std::string, double> > ValueVector;
  typedef std::map<std::string, std::vector<double> > VectorMap;

  // ______________________________________________________________________
  // input preprocessor (handles normalization and packing into Eigen)

  template<typename T>
  class InputPreprocessorT
  {
  public:
    InputPreprocessorT(const std::vector<Input>& inputs);
    VectorX<T> operator()(const ValueMap&) const;
  private:
    // input transformations
    VectorX<T> m_offsets;
    VectorX<T> m_scales;
    std::vector<std::string> m_names;
  };
  
  using InputPreprocessor = InputPreprocessorT<double>;

  template<typename T>
  class InputVectorPreprocessorT
  {
  public:
    InputVectorPreprocessorT(const std::vector<Input>& inputs);
    MatrixX<T> operator()(const VectorMap&) const;
  private:
    // input transformations
    VectorX<T> m_offsets;
    VectorX<T> m_scales;
    std::vector<std::string> m_names;
  };
  
  using InputVectorPreprocessor = InputVectorPreprocessorT<double>;
}

#include "InputPreprocessor.txx"

#endif
