#ifndef LWTNN_EIGEN_TYPEDEFS_HH
#define LWTNN_EIGEN_TYPEDEFS_HH

#include <Eigen/Dense>

namespace lwt {

  template<typename T>
  using VectorX = Eigen::Matrix<T, Eigen::Dynamic, 1>;

  template<typename T>
  using MatrixX = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

  template<typename T>
  using ArrayX = Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic>;

}

#endif // LWTNN_EIGEN_TYPEDEFS_HH
