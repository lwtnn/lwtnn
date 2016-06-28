#include "lwtnn/LightweightNeuralNetwork.hh"
#include <Eigen/Dense>

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
  UnaryActivationLayer::UnaryActivationLayer(Activation act):
    _func(get_activation(act))
  {
  }
  VectorXd UnaryActivationLayer::compute(const VectorXd& in) const {
    return in.unaryExpr(_func);
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

  // dense layer
  DenseLayer::DenseLayer(const MatrixXd& matrix,
                         const VectorXd& bias,
                         lwt::Activation activation):
  _matrix(matrix),
  _bias(bias),
  _activation(get_activation(activation))
  {
  }
  VectorXd DenseLayer::compute(const VectorXd& in) const {
    return (_matrix * in + _bias).unaryExpr(_activation);
  }

  // highway layer
  HighwayLayer::HighwayLayer(const MatrixXd& W,
                             const VectorXd& b,
                             const MatrixXd& W_carry,
                             const VectorXd& b_carry,
                             Activation activation):
    _w_t(W), _b_t(b), _w_c(W_carry), _b_c(b_carry),
    _act(get_activation(activation))
  {
  }
  VectorXd HighwayLayer::compute(const VectorXd& in) const {
    const std::function<double(double)> sig(nn_sigmoid);
    ArrayXd c = (_w_c * in + _b_c).unaryExpr(sig);
    ArrayXd t = (_w_t * in + _b_t).unaryExpr(_act);
    return c * t + (1 - c) * in.array();
  }

  // ______________________________________________________________________
  // Stack class

  // dummy construction routine
  Stack::Stack() {
    _layers.push_back(new DummyLayer);
    _layers.push_back(new UnaryActivationLayer(Activation::SIGMOID));
    _layers.push_back(new BiasLayer(std::vector<double>{1, 1, 1, 1}));
    MatrixXd mat(4, 4);
    mat <<
      0, 0, 0, 1,
      0, 0, 1, 0,
      0, 1, 0, 0,
      1, 0, 0, 0;
    _layers.push_back(new MatrixLayer(mat));
    _n_outputs = 4;
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
  // Private Stack methods to add various types of layers

  // top level add_layers method. This delegates to the other methods
  // below
  size_t Stack::add_layers(size_t n_inputs, const LayerConfig& layer) {
    if (layer.architecture == Architecture::DENSE) {
      return add_dense_layers(n_inputs, layer);
    } else if (layer.architecture == Architecture::HIGHWAY){
      return add_highway_layers(n_inputs, layer);
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


  size_t Stack::add_highway_layers(size_t n_inputs, const LayerConfig& layer) {
    auto& comps = layer.components;
    const auto& t = get_component(comps.at(Component::T), n_inputs);
    const auto& c = get_component(comps.at(Component::CARRY), n_inputs);

    _layers.push_back(
      new HighwayLayer(t.W, t.b, c.W, c.b, layer.activation));
    return n_inputs;
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

  // _____________________________________________________________________
  // Activation functions
  //
  // There are two functions below. In most cases the activation layer
  // can be implemented as a unary function, but in some cases
  // (i.e. softmax) something more complicated is reqired.

  // Note that in the first case you own this layer! It's your
  // responsibility to delete it.
  ILayer* get_raw_activation_layer(Activation activation) {
    // Check for special cases. If it's not one, use
    // UnaryActivationLayer
    switch (activation) {
    case Activation::SOFTMAX: return new SoftmaxLayer;
    default: return new UnaryActivationLayer(activation);
    }
  }

  // Most activation functions should be handled here.
  std::function<double(double)> get_activation(lwt::Activation act) {
    using namespace lwt;
    switch (act) {
    case Activation::SIGMOID: return nn_sigmoid;
    case Activation::HARD_SIGMOID: return nn_hard_sigmoid;
    case Activation::TANH: return nn_tanh;
    case Activation::RECTIFIED: return nn_relu;
    case Activation::LINEAR: return [](double x){return x;};
    default: {
      throw NNConfigurationException("Got undefined activation function");
    }
    }
  }


  double nn_sigmoid( double x ){
    //github.com/Theano/Theano/blob/master/theano/tensor/nnet/sigm.py#L35
    if (x < -30.0) return 0.0;
    if (x >  30.0) return 1.0;
    return 1.0 / (1.0 + std::exp(-1.0*x));
  }

  double nn_hard_sigmoid( double x ){
    //github.com/Theano/Theano/blob/master/theano/tensor/nnet/sigm.py#L279
    double out = 0.2*x + 0.5;
    if (out < 0) return 0.0;
    if (out > 1) return 1.0;
    return out;
  }

  double nn_tanh( double x ){
    return std::tanh(x);
  }

  double nn_relu( double x) {
    if (std::isnan(x)) return x;
    else return x > 0 ? x : 0;
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
    bool act_ok = layer.activation == Activation::NONE;
    if (wt_ok && bias_ok && maxout_ok && act_ok) return;
    throw NNConfigurationException("layer has wrong info for maxout");
  }
  void throw_if_not_dense(const LayerConfig& layer) {
    if (layer.sublayers.size() > 0) {
      throw NNConfigurationException("sublayers in dense layer");
    }
  }

  // component-wise getters (for Highway, lstm, etc)
  DenseComponents get_component(const lwt::LayerConfig& layer, size_t n_in) {
    using namespace Eigen;
    using namespace lwt;
    MatrixXd weights = build_matrix(layer.weights, n_in);
    size_t n_out = weights.rows();
    VectorXd bias = build_vector(layer.bias);

    // the u element is optional
    size_t u_el = layer.U.size();
    MatrixXd U = u_el ? build_matrix(layer.U, n_out) : MatrixXd::Zero(0,0);

    size_t u_out = U.rows();
    size_t b_out = bias.rows();
    bool u_mismatch = (u_out != n_out) && (u_out > 0);
    if ( u_mismatch || b_out != n_out) {
      throw NNConfigurationException(
        "Output dims mismatch, W: " + std::to_string(n_out) +
        ", U: " + std::to_string(u_out) + ", b: " + std::to_string(b_out));
    }
    return {weights, U, bias};
  }

}
