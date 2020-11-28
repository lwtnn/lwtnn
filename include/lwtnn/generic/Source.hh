#ifndef LWTNN_GENERIC_SOURCE_HH
#define LWTNN_GENERIC_SOURCE_HH

#include "lwtnn/generic/eigen_typedefs.hh"

#include <vector>

namespace lwt {
namespace generic {

  // this is called by input nodes to get the inputs
  template<typename T>
  class ISource
  {
  public:
    virtual ~ISource() = default;
    virtual VectorX<T> at(size_t index) const = 0;
    virtual MatrixX<T> matrix_at(size_t index) const = 0;
  };

  template<typename T>
  class VectorSource: public ISource<T>
  {
  public:
    VectorSource(std::vector<VectorX<T>>&&, std::vector<MatrixX<T>>&& = {});
    virtual VectorX<T> at(size_t index) const override;
    virtual MatrixX<T> matrix_at(size_t index) const override;
  private:
    std::vector<VectorX<T>> m_inputs;
    std::vector<MatrixX<T>> m_matrix_inputs;
  };

  template<typename T>
  class DummySource: public ISource<T>
  {
  public:
    DummySource(const std::vector<size_t>& input_sizes,
                const std::vector<std::pair<size_t, size_t> >& = {});
    virtual VectorX<T> at(size_t index) const override;
    virtual MatrixX<T> matrix_at(size_t index) const override;
  private:
    std::vector<size_t> m_sizes;
    std::vector<std::pair<size_t, size_t> > m_matrix_sizes;
  };

} // namespace generic

} // namespace lwt

#endif //SOURCE_HH
