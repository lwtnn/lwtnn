#include "LWTagger.hh"

namespace lwt {

  VectorXd DummyLayer::compute(const VectorXd& in) const {
    return in;
  }

  VectorXd SigmoidLayer::compute(const VectorXd& in) const {
    // todo: is there a more efficient way to do this?
    size_t n_elements = in.rows();
    VectorXd out(n_elements);
    for (size_t iii = 0; iii < n_elements; iii++) {
      out(iii) = 1 / (1 + std::exp(-in(iii)));
    }
    return out;
  }

  BiasLayer::BiasLayer(const VectorXd& bias): _bias(bias)
  {
  }
  BiasLayer::BiasLayer(const std::vector<double>& bias):
    _bias(bias.size())
  {
    size_t idx = 0;
    for (const auto& val: bias) {
      _bias(idx) = val;
      idx++;
    }
  }
  VectorXd BiasLayer::compute(const VectorXd& in) const {
    return in + _bias;
  }

  MatrixLayer::MatrixLayer(const MatrixXd& matrix):
    _matrix(matrix)
  {
  }
  VectorXd MatrixLayer::compute(const VectorXd& in) const {
    return _matrix * in;
  }

  // dummy construction routine
  Stack::Stack() {
    _layers.push_back(new DummyLayer);
    _layers.push_back(new SigmoidLayer);
    _layers.push_back(new BiasLayer(std::vector<double>{1, 1, 1, 1}));
    MatrixXd mat(4, 4);
    mat <<
      0, 0, 0, 1,
      0, 0, 1, 0,
      0, 1, 0, 0,
      1, 0, 0, 0;
    _layers.push_back(new MatrixLayer(mat));
  }
  Stack::~Stack() {
    for (auto& layer: _layers) {
      delete layer;
      layer = 0;
    }
  }
  VectorXd Stack::compute(VectorXd in) const {
    for (const auto& layer: _layers) {
      in = layer->compute(in);
    }
    return in;
  }
}
