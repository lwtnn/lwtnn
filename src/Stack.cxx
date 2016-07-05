#include "lwtnn/Stack.hh"
#include <Eigen/Dense>

#include <set>

// internal utility functions
namespace {
  using namespace Eigen;
  using namespace lwt;
}
namespace lwt {

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
  // Recurrent Stack

  RecurrentStack::RecurrentStack(size_t n_inputs,
                                 const std::vector<lwt::LayerConfig>& layers)
  {
    using namespace lwt;
    size_t layer_n = 0;
    const size_t n_layers = layers.size();
    for (;layer_n < n_layers; layer_n++) {
      auto& layer = layers.at(layer_n);

      // add recurrent layers (now LSTM and GRU!)
      if (layer.architecture == Architecture::LSTM) {
        n_inputs = add_lstm_layers(n_inputs, layer);
      } else if (layer.architecture == Architecture::GRU) {
        n_inputs = add_gru_layers(n_inputs, layer);
      } else if (layer.architecture == Architecture::EMBEDDING) {
        n_inputs = add_embedding_layers(n_inputs, layer);
      } else {
        // leave this loop if we're done with the recurrent stuff
        break;
      }
    }
    // fill the remaining dense layers
    _stack = new Stack(n_inputs, layers, layer_n);
  }
  RecurrentStack::~RecurrentStack() {
    for (auto& layer: _layers) {
      delete layer;
      layer = 0;
    }
    delete _stack;
    _stack = 0;
  }
  VectorXd RecurrentStack::reduce(MatrixXd in) const {
    for (auto* layer: _layers) {
      in = layer->scan(in);
    }
    return _stack->compute(in.col(in.cols() - 1));
  }
  size_t RecurrentStack::n_outputs() const {
    return _stack->n_outputs();
  }

  size_t RecurrentStack::add_lstm_layers(size_t n_inputs,
                                         const LayerConfig& layer) {
    auto& comps = layer.components;
    const auto& i = get_component(comps.at(Component::I), n_inputs);
    const auto& o = get_component(comps.at(Component::O), n_inputs);
    const auto& f = get_component(comps.at(Component::F), n_inputs);
    const auto& c = get_component(comps.at(Component::C), n_inputs);
    _layers.push_back(
      new LSTMLayer(layer.activation, layer.inner_activation,
                    i.W, i.U, i.b,
                    f.W, f.U, f.b,
                    o.W, o.U, o.b,
                    c.W, c.U, c.b));
    return o.b.rows();
  }

  size_t RecurrentStack::add_gru_layers(size_t n_inputs,
                                         const LayerConfig& layer) {
    auto& comps = layer.components;
    const auto& z = get_component(comps.at(Component::Z), n_inputs);
    const auto& r = get_component(comps.at(Component::R), n_inputs);
    const auto& h = get_component(comps.at(Component::H), n_inputs);
    _layers.push_back(
      new GRULayer(layer.activation, layer.inner_activation,
                    z.W, z.U, z.b,
                    r.W, r.U, r.b,
                    h.W, h.U, h.b));
    return h.b.rows();
  }

  size_t RecurrentStack::add_embedding_layers(size_t n_inputs,
                                              const LayerConfig& layer) {
    for (const auto& emb: layer.embedding) {
      size_t n_wt = emb.weights.size();
      size_t n_cats = n_wt / emb.n_out;
      MatrixXd mat = build_matrix(emb.weights, n_cats);
      _layers.push_back(new EmbeddingLayer(emb.index, mat));
      n_inputs += emb.n_out - 1;
    }
    return n_inputs;
  }

  EmbeddingLayer::EmbeddingLayer(int var_row_index, MatrixXd W):
    _var_row_index(var_row_index),
    _W(W)
  {
    if(var_row_index < 0)
      throw NNConfigurationException(
        "EmbeddingLayer::EmbeddingLayer - can not set var_row_index<0,"
        " it is an index for a matrix row!");
  }

  MatrixXd EmbeddingLayer::scan( const MatrixXd& x) {

    if( _var_row_index >= x.rows() )
      throw NNEvaluationException(
        "EmbeddingLayer::scan - var_row_index is larger than input matrix"
        " number of rows!");

    MatrixXd embedded(_W.rows(), x.cols());

    for(int icol=0; icol<x.cols(); icol++)
      embedded.col(icol) = _W.col( x(_var_row_index, icol) );

    //only embed 1 variable at a time, so this should be correct size
    MatrixXd out(_W.rows() + (x.rows() - 1), x.cols());

    //assuming _var_row_index is an index with first possible value of 0
    if(_var_row_index > 0)
      out.topRows(_var_row_index) = x.topRows(_var_row_index);

    out.block(_var_row_index, 0, embedded.rows(), embedded.cols()) = embedded;

    if( _var_row_index < (x.rows()-1) )
      out.bottomRows( x.cols() - 1 - _var_row_index)
        = x.bottomRows( x.cols() - 1 - _var_row_index);

    return out;
  }

  LSTMLayer::LSTMLayer(Activation activation, Activation inner_activation,
           MatrixXd W_i, MatrixXd U_i, VectorXd b_i,
           MatrixXd W_f, MatrixXd U_f, VectorXd b_f,
           MatrixXd W_o, MatrixXd U_o, VectorXd b_o,
           MatrixXd W_c, MatrixXd U_c, VectorXd b_c,
           bool return_sequences):
    _W_i(W_i),
    _U_i(U_i),
    _b_i(b_i),
    _W_f(W_f),
    _U_f(U_f),
    _b_f(b_f),
    _W_o(W_o),
    _U_o(U_o),
    _b_o(b_o),
    _W_c(W_c),
    _U_c(U_c),
    _b_c(b_c),
    _time(-1),
    _return_sequences(return_sequences)
  {
    _n_outputs = _W_o.rows();

    _activation_fun = get_activation(activation);
    _inner_activation_fun = get_activation(inner_activation);
  }

  VectorXd LSTMLayer::step( const VectorXd& x_t ) {
    // https://github.com/fchollet/keras/blob/master/keras/layers/recurrent.py#L740

    if(_time < 0)
      throw NNEvaluationException(
        "LSTMLayer::compute - time is less than zero!");

    const auto& act_fun = _activation_fun;
    const auto& in_act_fun = _inner_activation_fun;

    int tm1 = std::max(0, _time - 1);
    VectorXd h_tm1 = _h_t.col(tm1);
    VectorXd C_tm1 = _C_t.col(tm1);

    VectorXd i  =  (_W_i*x_t + _b_i + _U_i*h_tm1).unaryExpr(in_act_fun);
    VectorXd f  =  (_W_f*x_t + _b_f + _U_f*h_tm1).unaryExpr(in_act_fun);
    VectorXd o  =  (_W_o*x_t + _b_o + _U_o*h_tm1).unaryExpr(in_act_fun);
    VectorXd ct =  (_W_c*x_t + _b_c + _U_c*h_tm1).unaryExpr(act_fun);

    _C_t.col(_time) = f.cwiseProduct(C_tm1) + i.cwiseProduct(ct);
    _h_t.col(_time) = o.cwiseProduct( _C_t.col(_time).unaryExpr(act_fun) );

    return VectorXd( _h_t.col(_time) );
  }

  MatrixXd LSTMLayer::scan( const MatrixXd& x ){

    _C_t.resize(_n_outputs, x.cols());
    _C_t.setZero();
    _h_t.resize(_n_outputs, x.cols());
    _h_t.setZero();
    _time = -1;


    for(_time=0; _time < x.cols(); _time++) {
      this->step( x.col( _time ) );
    }

    return _return_sequences ? _h_t : _h_t.col(_h_t.cols() - 1);
  }


  GRULayer::GRULayer(Activation activation, Activation inner_activation,
           MatrixXd W_z, MatrixXd U_z, VectorXd b_z,
           MatrixXd W_r, MatrixXd U_r, VectorXd b_r,
           MatrixXd W_h, MatrixXd U_h, VectorXd b_h,
           bool return_sequences):
    _W_z(W_z),
    _U_z(U_z),
    _b_z(b_z),
    _W_r(W_r),
    _U_r(U_r),
    _b_r(b_r),
    _W_h(W_h),
    _U_h(U_h),
    _b_h(b_h),
    _time(-1),
    _return_sequences(return_sequences)
  {
    _n_outputs = _W_h.rows();

    _activation_fun = get_activation(activation);
    _inner_activation_fun = get_activation(inner_activation);
  }

  VectorXd GRULayer::step( const VectorXd& x_t ) {
    // https://github.com/fchollet/keras/blob/master/keras/layers/recurrent.py#L547

    if(_time < 0)
      throw NNEvaluationException(
        "LSTMLayer::compute - time is less than zero!");

    const auto& act_fun = _activation_fun;
    const auto& in_act_fun = _inner_activation_fun;

    int tm1 = std::max(0, _time - 1);
    VectorXd h_tm1 = _h_t.col(tm1);
    //VectorXd C_tm1 = _C_t.col(tm1);
    VectorXd z  = (_W_z*x_t + _b_z + _U_z*h_tm1).unaryExpr(in_act_fun);
    VectorXd r  = (_W_r*x_t + _b_r + _U_r*h_tm1).unaryExpr(in_act_fun);
    VectorXd hh = (_W_h*x_t + _b_h + _U_h*(r.cwiseProduct(h_tm1))).unaryExpr(act_fun); 
    _h_t.col(_time)  = z.cwiseProduct(h_tm1) + (VectorXd::Ones(z.size()) - z).cwiseProduct(hh);

    return VectorXd( _h_t.col(_time) );
  }

  MatrixXd GRULayer::scan( const MatrixXd& x ){

    _h_t.resize(_n_outputs, x.cols());
    _h_t.setZero();
    _time = -1;

    for(_time=0; _time < x.cols(); _time++){
  this->step( x.col( _time ) );
      }

    return _return_sequences ? _h_t : _h_t.col(_h_t.cols() - 1);
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
