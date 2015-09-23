#include "LWTagger.hh"

namespace lwt {

  VectorXd DummyLayer::compute(const VectorXd& in) const {
    return in;
  }

  VectorXd SigmoidLayer::compute(const VectorXd& in) const {
    // TODO: is there a more efficient way to do this?
    size_t n_elements = in.rows();
    VectorXd out(n_elements);
    for (size_t iii = 0; iii < n_elements; iii++) {
      out(iii) = 1 / (1 + std::exp(-in(iii)));
    }
    return out;
  }
  VectorXd RectifiedLayer::compute(const VectorXd& in) const {
    // TODO: is there a more efficient way to do this?
    size_t n_elements = in.rows();
    VectorXd out(n_elements);
    for (size_t iii = 0; iii < n_elements; iii++) {
      out(iii) = in(iii) > 0 ? in(iii) : 0;
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

  // ______________________________________________________________________
  // Stack class

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

  // construct from LayerConfig
  Stack::Stack(size_t n_inputs, const std::vector<LayerConfig>& layers) {
    for (const auto& layer: layers) {
      n_inputs = add_layers(n_inputs, layer);
    }
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

  // _______________________________________________________________________
  // private stuff
  size_t Stack::add_layers(size_t n_inputs, const LayerConfig& layer) {
    size_t n_outputs = n_inputs;

    // add matrix layer
    if (layer.weights.size() > 0) {
      size_t n_elements = layer.weights.size();
      if ((n_elements % n_inputs) != 0) {
	std::string problem = "matrix elements not divisible by number"
	  " of columns. Elements: " + std::to_string(n_elements) +
	  ", Inputs: " + std::to_string(n_inputs);
	throw NNConfigurationException(problem);
      }
      n_outputs = n_elements / n_inputs;
      MatrixXd matrix(n_outputs, n_inputs);
      for (size_t row = 0; row < n_outputs; row++) {
	for (size_t col = 0; col < n_inputs; col++) {
	  double element = layer.weights.at(col + row * n_inputs);
	  matrix(row, col) = element;
	}
      }
      _layers.push_back(new MatrixLayer(matrix));
    };

    // add bias layer
    if (layer.bias.size() > 0) {
      if (n_outputs != layer.bias.size() ) {
	std::string problem = "tried to add a bias layer with " +
	  std::to_string(layer.bias.size()) + " entries, previous layer"
	  " had " + std::to_string(n_outputs) + " outputs";
	throw NNConfigurationException(problem);
      }
      _layers.push_back(new BiasLayer(layer.bias));
    }
    // add activation layer
    if (layer.activation != Activation::LINEAR) {
      _layers.push_back(get_raw_activation_layer(layer.activation));
    }
    return n_outputs;
  }

  // function for internal use
  ILayer* get_raw_activation_layer(Activation activation) {
    switch (activation) {
    case Activation::SIGMOID: return new SigmoidLayer;
    case Activation::RECTIFIED: return new RectifiedLayer;
    default: {
      std::string problem = "asked for a non-implemented activation function";
      throw NNConfigurationException(problem);
    }
    }
  }

  // ______________________________________________________________________
  // excpetions
  NNConfigurationException::NNConfigurationException(std::string problem):
    std::logic_error(problem)
  {}
}
