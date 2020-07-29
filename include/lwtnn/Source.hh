#ifndef SOURCE_HH
#define SOURCE_HH

#include <Eigen/Dense>

#include <vector>

namespace lwt {
    
  template<typename T>
  using VectorX = Eigen::Matrix<T, Eigen::Dynamic, 1>;
  
  template<typename T>
  using MatrixX = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

  // this is called by input nodes to get the inputs
  template<typename T>
  class ISourceT
  {
  public:
    virtual ~ISourceT() = default;
    virtual VectorX<T> at(size_t index) const = 0;
    virtual MatrixX<T> matrix_at(size_t index) const = 0;
  };
  
  using ISource = ISourceT<double>;

  template<typename T>
  class VectorSourceT: public ISourceT<T>
  {
  public:
    VectorSourceT(std::vector<VectorX<T>>&&, std::vector<MatrixX<T>>&& = {});
    virtual VectorX<T> at(size_t index) const override;
    virtual MatrixX<T> matrix_at(size_t index) const override;
  private:
    std::vector<VectorX<T>> m_inputs;
    std::vector<MatrixX<T>> m_matrix_inputs;
  };
  
  using VectorSource = VectorSourceT<double>;

  template<typename T>
  class DummySourceT: public ISourceT<T>
  {
  public:
    DummySourceT(const std::vector<size_t>& input_sizes,
                const std::vector<std::pair<size_t, size_t> >& = {});
    virtual VectorX<T> at(size_t index) const override;
    virtual MatrixX<T> matrix_at(size_t index) const override;
  private:
    std::vector<size_t> m_sizes;
    std::vector<std::pair<size_t, size_t> > m_matrix_sizes;
  };
  
  using DummySource = DummySourceT<double>;
}

#endif //SOURCE_HH
