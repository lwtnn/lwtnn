#include "LightweightNeuralNetwork.hh"

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
      // pass through NaN values
      if (std::isnan(in(iii))) out(iii) = in(iii);
      else out(iii) = in(iii) > 0 ? in(iii) : 0;
    }
    return out;
  }
  VectorXd SoftmaxLayer::compute(const VectorXd& in) const {
    size_t n_elements = in.rows();
    VectorXd exp(n_elements);
    for (size_t iii = 0; iii < n_elements; iii++) {
      exp(iii) = std::exp(in(iii));
    }
    double sum_exp = exp.sum();
    return exp / sum_exp;
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
    // the final assigned n_inputs is the number of output nodes
    _n_outputs = n_inputs;
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
  size_t Stack::n_outputs() const {
    return _n_outputs;
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
    case Activation::SOFTMAX: return new SoftmaxLayer;
    default: {
      std::string problem = "asked for a non-implemented activation function";
      throw NNConfigurationException(problem);
    }
    }
  }

  // ______________________________________________________________________
  // LightweightNeuralNetwork HL wrapper
  LightweightNeuralNetwork::LightweightNeuralNetwork(
    const std::vector<Input>& inputs,
    const std::vector<LayerConfig>& layers,
    const std::vector<std::string>& outputs):
    _stack(inputs.size(), layers),
    _offsets(inputs.size()),
    _scales(inputs.size()),
    _outputs(outputs.begin(), outputs.end())
  {
    size_t in_num = 0;
    for (const auto& input: inputs) {
      _offsets(in_num) = input.offset;
      _scales(in_num) = input.scale;
      _names.push_back(input.name);
      in_num++;
    }
    if (_outputs.size() != _stack.n_outputs()) {
      std::string problem = "internal stack has " +
        std::to_string(_stack.n_outputs()) + " outputs, but " +
        std::to_string(_outputs.size()) + " were given";
      throw NNConfigurationException(problem);
    }
  }

  LightweightNeuralNetwork::ValueMap LightweightNeuralNetwork::compute(
    const LightweightNeuralNetwork::ValueMap& in) const {
    VectorXd invec(_names.size());
    size_t input_number = 0;
    for (const auto& in_name: _names) {
      if (!in.count(in_name)) {
        throw NNEvaluationException("can't find input: " + in_name);
      }
      invec(input_number) = in.at(in_name);
      input_number++;
    }

    // compute outputs
    auto outvec = _stack.compute((invec + _offsets).cwiseProduct(_scales));
    assert(outvec.rows() > 0);
    auto out_size = static_cast<size_t>(outvec.rows());
    assert(out_size == _outputs.size());

    // build and return output map
    LightweightNeuralNetwork::ValueMap out_map;
    for (size_t out_n = 0; out_n < out_size; out_n++) {
      out_map.emplace(_outputs.at(out_n), outvec(out_n));
    }
    return out_map;
  }


  // ______________________________________________________________________
  // excpetions
  LightweightNNException::LightweightNNException(std::string problem):
    std::logic_error(problem)
  {}
  NNConfigurationException::NNConfigurationException(std::string problem):
    LightweightNNException(problem)
  {}
  NNEvaluationException::NNEvaluationException(std::string problem):
    LightweightNNException(problem)
  {}
}
