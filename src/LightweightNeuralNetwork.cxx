#include "lwtnn/LightweightNeuralNetwork.hh"

#include <set>

// internal utility functions
namespace {
  using namespace Eigen;
  using namespace lwt;
}
namespace lwt {

  // _______________________________________________________________________
  // layer implementation

  VectorXd DummyLayer::compute(const VectorXd& in) const {
    return in;
  }

  // activation functions
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
  VectorXd TanhLayer::compute(const VectorXd& in) const {
    size_t n_elements = in.rows();
    VectorXd out(n_elements);
    for (size_t iii = 0; iii < n_elements; iii++) {
      out(iii) = std::tanh(in(iii));
    }
    return out;
  }

  // bias layer
  BiasLayer::BiasLayer(const VectorXd& bias): _bias(bias)
  {
  }
  BiasLayer::BiasLayer(const std::vector<double>& bias):
    _bias(build_vector(bias))
  {
  }
  VectorXd BiasLayer::compute(const VectorXd& in) const {
    return in + _bias;
  }

  // basic dense matrix layer
  MatrixLayer::MatrixLayer(const MatrixXd& matrix):
    _matrix(matrix)
  {
  }
  VectorXd MatrixLayer::compute(const VectorXd& in) const {
    return _matrix * in;
  }

  // maxout layer
  MaxoutLayer::MaxoutLayer(const std::vector<MaxoutLayer::InitUnit>& units):
    _bias(units.size(), units.front().first.rows())
  {
    int out_pos = 0;
    for (const auto& unit: units) {
      _matrices.push_back(unit.first);
      _bias.row(out_pos) = unit.second;
      out_pos++;
    }
  }
  VectorXd MaxoutLayer::compute(const VectorXd& in) const {
    // eigen supports tensors, but only in the experimental component
    // for now just stick to matrix and vector classes
    const size_t n_mat = _matrices.size();
    const size_t out_dim = _matrices.front().rows();
    MatrixXd outputs(n_mat, out_dim);
    for (size_t mat_n = 0; mat_n < n_mat; mat_n++) {
      outputs.row(mat_n) = _matrices.at(mat_n) * in;
    }
    outputs += _bias;
    return outputs.colwise().maxCoeff();
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
  Stack::Stack(size_t n_inputs, const std::vector<LayerConfig>& layers,
               size_t skip) {
    for (size_t nnn = skip; nnn < layers.size(); nnn++) {
      n_inputs = add_layers(n_inputs, layers.at(nnn));
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
    if (layer.architecture == Architecture::DENSE) {
      return add_dense_layers(n_inputs, layer);
    } else if (layer.architecture == Architecture::MAXOUT) {
      return add_maxout_layers(n_inputs, layer);
    }
    throw NNConfigurationException("unknown architecture");
  }
  size_t Stack::add_dense_layers(size_t n_inputs, const LayerConfig& layer) {
    assert(layer.architecture == Architecture::DENSE);
    throw_if_not_dense(layer);

    size_t n_outputs = n_inputs;

    // add matrix layer
    if (layer.weights.size() > 0) {
      MatrixXd matrix = build_matrix(layer.weights, n_inputs);
      n_outputs = matrix.rows();
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
  size_t Stack::add_maxout_layers(size_t n_inputs, const LayerConfig& layer) {
    assert(layer.architecture == Architecture::MAXOUT);
    throw_if_not_maxout(layer);
    std::vector<MaxoutLayer::InitUnit> matrices;
    std::set<size_t> n_outputs;
    for (const auto& sublayer: layer.sublayers) {
      MatrixXd matrix = build_matrix(sublayer.weights, n_inputs);
      VectorXd bias = build_vector(sublayer.bias);
      n_outputs.insert(matrix.rows());
      matrices.push_back(std::make_pair(matrix, bias));
    }
    if (n_outputs.size() == 0) {
      throw NNConfigurationException("tried to build maxout withoutweights!");
    }
    else if (n_outputs.size() != 1) {
      throw NNConfigurationException("uneven matrices for maxout");
    }
    _layers.push_back(new MaxoutLayer(matrices));
    return *n_outputs.begin();
  }

  // function for internal use
  ILayer* get_raw_activation_layer(Activation activation) {
    switch (activation) {
    case Activation::SIGMOID: return new SigmoidLayer;
    case Activation::RECTIFIED: return new RectifiedLayer;
    case Activation::SOFTMAX: return new SoftmaxLayer;
    case Activation::TANH: return new TanhLayer;
    default: {
      std::string problem = "asked for a non-implemented activation function";
      throw NNConfigurationException(problem);
    }
    }
  }

  // ______________________________________________________________________
  // Input preprocessor
  InputPreprocessor::InputPreprocessor(const std::vector<Input>& inputs):
    _offsets(inputs.size()),
    _scales(inputs.size())
  {
    size_t in_num = 0;
    for (const auto& input: inputs) {
      _offsets(in_num) = input.offset;
      _scales(in_num) = input.scale;
      _names.push_back(input.name);
      in_num++;
    }
  }
  VectorXd InputPreprocessor::operator()(const ValueMap& in) const {
    VectorXd invec(_names.size());
    size_t input_number = 0;
    for (const auto& in_name: _names) {
      if (!in.count(in_name)) {
        throw NNEvaluationException("can't find input: " + in_name);
      }
      invec(input_number) = in.at(in_name);
      input_number++;
    }
    return (invec + _offsets).cwiseProduct(_scales);
  }

  // ______________________________________________________________________
  // LightweightNeuralNetwork HL wrapper
  LightweightNeuralNetwork::LightweightNeuralNetwork(
    const std::vector<Input>& inputs,
    const std::vector<LayerConfig>& layers,
    const std::vector<std::string>& outputs):
    _stack(inputs.size(), layers),
    _preproc(inputs),
    _outputs(outputs.begin(), outputs.end())
  {
    if (_outputs.size() != _stack.n_outputs()) {
      std::string problem = "internal stack has " +
        std::to_string(_stack.n_outputs()) + " outputs, but " +
        std::to_string(_outputs.size()) + " were given";
      throw NNConfigurationException(problem);
    }
  }

  lwt::ValueMap
  LightweightNeuralNetwork::compute(const lwt::ValueMap& in) const {

    // compute outputs
    auto outvec = _stack.compute(_preproc(in));
    assert(outvec.rows() > 0);
    auto out_size = static_cast<size_t>(outvec.rows());
    assert(out_size == _outputs.size());

    // build and return output map
    lwt::ValueMap out_map;
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

// ________________________________________________________________________
// utility functions
  MatrixXd build_matrix(const std::vector<double>& weights, size_t n_inputs)
  {
    size_t n_elements = weights.size();
    if ((n_elements % n_inputs) != 0) {
      std::string problem = "matrix elements not divisible by number"
        " of columns. Elements: " + std::to_string(n_elements) +
        ", Inputs: " + std::to_string(n_inputs);
      throw lwt::NNConfigurationException(problem);
    }
    size_t n_outputs = n_elements / n_inputs;
    MatrixXd matrix(n_outputs, n_inputs);
    for (size_t row = 0; row < n_outputs; row++) {
      for (size_t col = 0; col < n_inputs; col++) {
        double element = weights.at(col + row * n_inputs);
        matrix(row, col) = element;
      }
    }
    return matrix;
  }
  VectorXd build_vector(const std::vector<double>& bias) {
    VectorXd out(bias.size());
    size_t idx = 0;
    for (const auto& val: bias) {
      out(idx) = val;
      idx++;
    }
    return out;
  }

  // consistency checks
  void throw_if_not_maxout(const LayerConfig& layer) {
    bool wt_ok = layer.weights.size() == 0;
    bool bias_ok = layer.bias.size() == 0;
    bool maxout_ok = layer.sublayers.size() > 0;
    bool act_ok = layer.activation == Activation::LINEAR;
    if (wt_ok && bias_ok && maxout_ok && act_ok) return;
    throw NNConfigurationException("layer has wrong info for maxout");
  }
  void throw_if_not_dense(const LayerConfig& layer) {
    if (layer.sublayers.size() > 0) {
      throw NNConfigurationException("sublayers in dense layer");
    }
  }

}
