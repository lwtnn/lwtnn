#ifndef SOURCE_HH
#define SOURCE_HH

#include <Eigen/Dense>

#include <vector>

namespace lwt {
  using Eigen::VectorXd;
  using Eigen::MatrixXd;

  // this is called by input nodes to get the inputs
  class ISource
  {
  public:
    virtual VectorXd at(size_t index) const = 0;
    virtual MatrixXd matrix_at(size_t index) const = 0;
  };

  class VectorSource: public ISource
  {
  public:
    VectorSource(std::vector<VectorXd>&&, std::vector<MatrixXd>&& = {});
    virtual VectorXd at(size_t index) const;
    virtual MatrixXd matrix_at(size_t index) const;
  private:
    std::vector<VectorXd> m_inputs;
    std::vector<MatrixXd> m_matrix_inputs;
  };

  class DummySource: public ISource
  {
  public:
    DummySource(const std::vector<size_t>& input_sizes,
                const std::vector<std::pair<size_t, size_t> >& = {});
    virtual VectorXd at(size_t index) const;
    virtual MatrixXd matrix_at(size_t index) const;
  private:
    std::vector<size_t> m_sizes;
    std::vector<std::pair<size_t, size_t> > m_matrix_sizes;
  };
}

#endif //SOURCE_HH
