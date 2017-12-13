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
  // Feed forward Stack class

  // dummy construction routine
  Stack::Stack() {
    m_layers.push_back(new DummyLayer);
    m_layers.push_back(new UnaryActivationLayer(Activation::SIGMOID));
    m_layers.push_back(new BiasLayer(std::vector<double>{1, 1, 1, 1}));
    MatrixXd mat(4, 4);
    mat <<
      0, 0, 0, 1,
      0, 0, 1, 0,
      0, 1, 0, 0,
      1, 0, 0, 0;
    m_layers.push_back(new MatrixLayer(mat));
    m_n_outputs = 4;
  }

  // construct from LayerConfig
  Stack::Stack(size_t n_inputs, const std::vector<LayerConfig>& layers,
               size_t skip) {
    for (size_t nnn = skip; nnn < layers.size(); nnn++) {
      n_inputs = add_layers(n_inputs, layers.at(nnn));
    }
    // the final assigned n_inputs is the number of output nodes
    m_n_outputs = n_inputs;
  }

  Stack::~Stack() {
    for (auto& layer: m_layers) {
      delete layer;
      layer = 0;
    }
  }
  VectorXd Stack::compute(VectorXd in) const {
    for (const auto& layer: m_layers) {
      in = layer->compute(in);
    }
    return in;
  }
  size_t Stack::n_outputs() const {
    return m_n_outputs;
  }


  // Private Stack methods to add various types of layers
  //
  // top level add_layers method. This delegates to the other methods
  // below
  size_t Stack::add_layers(size_t n_inputs, const LayerConfig& layer) {
    if (layer.architecture == Architecture::DENSE) {
      return add_dense_layers(n_inputs, layer);
    } else if (layer.architecture == Architecture::NORMALIZATION){
      return add_normalization_layers(n_inputs, layer);
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
      m_layers.push_back(new MatrixLayer(matrix));
    };

    // add bias layer
    if (layer.bias.size() > 0) {
      if (n_outputs != layer.bias.size() ) {
        std::string problem = "tried to add a bias layer with " +
          std::to_string(layer.bias.size()) + " entries, previous layer"
          " had " + std::to_string(n_outputs) + " outputs";
        throw NNConfigurationException(problem);
      }
      m_layers.push_back(new BiasLayer(layer.bias));
    }

    // add activation layer
    if (layer.activation != Activation::LINEAR) {
      m_layers.push_back(get_raw_activation_layer(layer.activation));
    }

    return n_outputs;
  }

  size_t Stack::add_normalization_layers(size_t n_inputs, const LayerConfig& layer) {
    assert(layer.architecture == Architecture::NORMALIZATION);
    throw_if_not_normalization(layer);

    // Do some checks
    if ( layer.weights.size() < 1 || layer.bias.size() < 1 ) {
      std::string problem = "Either weights or bias layer size is < 1";
      throw NNConfigurationException(problem);
    };
    if ( layer.weights.size() != layer.bias.size() ) {
      std::string problem = "weights and bias layer are not equal in size!";
      throw NNConfigurationException(problem);
    };
    VectorXd v_weights = build_vector(layer.weights);
    VectorXd v_bias = build_vector(layer.bias);

    m_layers.push_back(
      new NormalizationLayer(v_weights, v_bias));
    return n_inputs;
  }


  size_t Stack::add_highway_layers(size_t n_inputs, const LayerConfig& layer) {
    auto& comps = layer.components;
    const auto& t = get_component(comps.at(Component::T), n_inputs);
    const auto& c = get_component(comps.at(Component::CARRY), n_inputs);

    m_layers.push_back(
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
    m_layers.push_back(new MaxoutLayer(matrices));
    return *n_outputs.begin();
  }


  // _______________________________________________________________________
  // Feed-forward layers

  VectorXd DummyLayer::compute(const VectorXd& in) const {
    return in;
  }

  // activation functions
  UnaryActivationLayer::UnaryActivationLayer(Activation act):
    m_func(get_activation(act))
  {
  }
  VectorXd UnaryActivationLayer::compute(const VectorXd& in) const {
    return in.unaryExpr(m_func);
  }

  VectorXd SoftmaxLayer::compute(const VectorXd& in) const {
    // More numerically stable softmax, as suggested in
    // http://stackoverflow.com/a/34969389
    size_t n_elements = in.rows();
    VectorXd exp(n_elements);
    double max = in.maxCoeff();
    for (size_t iii = 0; iii < n_elements; iii++) {
      exp(iii) = std::exp(in(iii) - max);
    }
    double sum_exp = exp.sum();
    return exp / sum_exp;
  }

  // bias layer
  BiasLayer::BiasLayer(const VectorXd& bias): m_bias(bias)
  {
  }
  BiasLayer::BiasLayer(const std::vector<double>& bias):
    m_bias(build_vector(bias))
  {
  }
  VectorXd BiasLayer::compute(const VectorXd& in) const {
    return in + m_bias;
  }

  // basic dense matrix layer
  MatrixLayer::MatrixLayer(const MatrixXd& matrix):
    m_matrix(matrix)
  {
  }
  VectorXd MatrixLayer::compute(const VectorXd& in) const {
    return m_matrix * in;
  }

  // maxout layer
  MaxoutLayer::MaxoutLayer(const std::vector<MaxoutLayer::InitUnit>& units):
    m_bias(units.size(), units.front().first.rows())
  {
    int out_pos = 0;
    for (const auto& unit: units) {
      m_matrices.push_back(unit.first);
      m_bias.row(out_pos) = unit.second;
      out_pos++;
    }
  }
  VectorXd MaxoutLayer::compute(const VectorXd& in) const {
    // eigen supports tensors, but only in the experimental component
    // for now just stick to matrix and vector classes
    const size_t n_mat = m_matrices.size();
    const size_t out_dim = m_matrices.front().rows();
    MatrixXd outputs(n_mat, out_dim);
    for (size_t mat_n = 0; mat_n < n_mat; mat_n++) {
      outputs.row(mat_n) = m_matrices.at(mat_n) * in;
    }
    outputs += m_bias;
    return outputs.colwise().maxCoeff();
  }

   // Normalization layer
   NormalizationLayer::NormalizationLayer(const VectorXd& W,
                                          const VectorXd& b):
    _W(W), _b(b)
  {
  }
  VectorXd NormalizationLayer::compute(const VectorXd& in) const {
    VectorXd shift = in + _b ;
    return _W.cwiseProduct(shift);
  }

  // highway layer
  HighwayLayer::HighwayLayer(const MatrixXd& W,
                             const VectorXd& b,
                             const MatrixXd& W_carry,
                             const VectorXd& b_carry,
                             Activation activation):
    m_w_t(W), m_b_t(b), m_w_c(W_carry), m_b_c(b_carry),
    m_act(get_activation(activation))
  {
  }
  VectorXd HighwayLayer::compute(const VectorXd& in) const {
    const std::function<double(double)> sig(nn_sigmoid);
    ArrayXd c = (m_w_c * in + m_b_c).unaryExpr(sig);
    ArrayXd t = (m_w_t * in + m_b_t).unaryExpr(m_act);
    return c * t + (1 - c) * in.array();
  }

  // ______________________________________________________________________
  // Recurrent Stack

  RecurrentStack::RecurrentStack(size_t n_inputs,
                                 const std::vector<lwt::LayerConfig>& layers)
  {
    using namespace lwt;
    const size_t n_layers = layers.size();
    for (size_t layer_n = 0; layer_n < n_layers; layer_n++) {
      auto& layer = layers.at(layer_n);

      // add recurrent layers (now LSTM and GRU!)
      if (layer.architecture == Architecture::LSTM) {
        n_inputs = add_lstm_layers(n_inputs, layer);
      } else if (layer.architecture == Architecture::GRU) {
        n_inputs = add_gru_layers(n_inputs, layer);
      } else if (layer.architecture == Architecture::EMBEDDING) {
        n_inputs = add_embedding_layers(n_inputs, layer);
      } else {
        throw NNConfigurationException("found non-recurrent layer");
      }
    }
    m_n_outputs = n_inputs;
  }
  RecurrentStack::~RecurrentStack() {
    for (auto& layer: m_layers) {
      delete layer;
      layer = 0;
    }
  }
  MatrixXd RecurrentStack::scan(MatrixXd in) const {
    for (auto* layer: m_layers) {
      in = layer->scan(in);
    }
    return in;
  }
  size_t RecurrentStack::n_outputs() const {
    return m_n_outputs;
  }

  size_t RecurrentStack::add_lstm_layers(size_t n_inputs,
                                         const LayerConfig& layer) {
    auto& comps = layer.components;
    const auto& i = get_component(comps.at(Component::I), n_inputs);
    const auto& o = get_component(comps.at(Component::O), n_inputs);
    const auto& f = get_component(comps.at(Component::F), n_inputs);
    const auto& c = get_component(comps.at(Component::C), n_inputs);
    m_layers.push_back(
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
    m_layers.push_back(
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
      m_layers.push_back(new EmbeddingLayer(emb.index, mat));
      n_inputs += emb.n_out - 1;
    }
    return n_inputs;
  }

  ReductionStack::ReductionStack(size_t n_in,
                                 const std::vector<LayerConfig>& layers) {
    std::vector<LayerConfig> recurrent;
    std::vector<LayerConfig> feed_forward;
    std::set<Architecture> recurrent_arcs{
      Architecture::LSTM, Architecture::GRU, Architecture::EMBEDDING};
    for (const auto& layer: layers) {
      if (recurrent_arcs.count(layer.architecture)) {
        recurrent.push_back(layer);
      } else {
        feed_forward.push_back(layer);
      }
    }
    m_recurrent = new RecurrentStack(n_in, recurrent);
    m_stack = new Stack(m_recurrent->n_outputs(), feed_forward);
  }
  ReductionStack::~ReductionStack() {
    delete m_recurrent;
    delete m_stack;
  }
  VectorXd ReductionStack::reduce(MatrixXd in) const {
    in = m_recurrent->scan(in);
    return m_stack->compute(in.col(in.cols() -1));
  }
  size_t ReductionStack::n_outputs() const {
    return m_stack->n_outputs();
  }

  // __________________________________________________________________
  // Recurrent layers

  EmbeddingLayer::EmbeddingLayer(int var_row_index, MatrixXd W):
    m_var_row_index(var_row_index),
    m_W(W)
  {
    if(var_row_index < 0)
      throw NNConfigurationException(
        "EmbeddingLayer::EmbeddingLayer - can not set var_row_index<0,"
        " it is an index for a matrix row!");
  }

  MatrixXd EmbeddingLayer::scan( const MatrixXd& x) const {

    if( m_var_row_index >= x.rows() )
      throw NNEvaluationException(
        "EmbeddingLayer::scan - var_row_index is larger than input matrix"
        " number of rows!");

    MatrixXd embedded(m_W.rows(), x.cols());

    for(int icol=0; icol<x.cols(); icol++) {
      double vector_idx = x(m_var_row_index, icol);
      bool is_int = std::floor(vector_idx) == vector_idx;
      bool is_valid = (vector_idx >= 0) && (vector_idx < m_W.cols());
      if (!is_int || !is_valid) throw NNEvaluationException(
        "Invalid embedded index: " + std::to_string(vector_idx));
      embedded.col(icol) = m_W.col( vector_idx );
    }

    //only embed 1 variable at a time, so this should be correct size
    MatrixXd out(m_W.rows() + (x.rows() - 1), x.cols());

    //assuming m_var_row_index is an index with first possible value of 0
    if(m_var_row_index > 0)
      out.topRows(m_var_row_index) = x.topRows(m_var_row_index);

    out.block(m_var_row_index, 0, embedded.rows(), embedded.cols()) = embedded;

    if( m_var_row_index < (x.rows()-1) )
      out.bottomRows( x.cols() - 1 - m_var_row_index)
        = x.bottomRows( x.cols() - 1 - m_var_row_index);

    return out;
  }


  // LSTM layer
  LSTMLayer::LSTMLayer(Activation activation, Activation inner_activation,
           MatrixXd W_i, MatrixXd U_i, VectorXd b_i,
           MatrixXd W_f, MatrixXd U_f, VectorXd b_f,
           MatrixXd W_o, MatrixXd U_o, VectorXd b_o,
           MatrixXd W_c, MatrixXd U_c, VectorXd b_c):
    m_W_i(W_i),
    m_U_i(U_i),
    m_b_i(b_i),
    m_W_f(W_f),
    m_U_f(U_f),
    m_b_f(b_f),
    m_W_o(W_o),
    m_U_o(U_o),
    m_b_o(b_o),
    m_W_c(W_c),
    m_U_c(U_c),
    m_b_c(b_c)
  {
    m_n_outputs = m_W_o.rows();

    m_activation_fun = get_activation(activation);
    m_inner_activation_fun = get_activation(inner_activation);
  }

  // internal structure created on each scan call
  struct LSTMState {
    LSTMState(size_t n_input, size_t n_outputs);
    MatrixXd C_t;
    MatrixXd h_t;
    int time;
  };
  LSTMState::LSTMState(size_t n_input, size_t n_output):
    C_t(MatrixXd::Zero(n_output, n_input)),
    h_t(MatrixXd::Zero(n_output, n_input)),
    time(0)
  {
  }

  void LSTMLayer::step(const VectorXd& x_t, LSTMState& s) const {
    // https://github.com/fchollet/keras/blob/master/keras/layers/recurrent.py#L740

    const auto& act_fun = m_activation_fun;
    const auto& in_act_fun = m_inner_activation_fun;

    int tm1 = s.time == 0 ? 0 : s.time - 1;
    VectorXd h_tm1 = s.h_t.col(tm1);
    VectorXd C_tm1 = s.C_t.col(tm1);

    VectorXd i  =  (m_W_i*x_t + m_b_i + m_U_i*h_tm1).unaryExpr(in_act_fun);
    VectorXd f  =  (m_W_f*x_t + m_b_f + m_U_f*h_tm1).unaryExpr(in_act_fun);
    VectorXd o  =  (m_W_o*x_t + m_b_o + m_U_o*h_tm1).unaryExpr(in_act_fun);
    VectorXd ct =  (m_W_c*x_t + m_b_c + m_U_c*h_tm1).unaryExpr(act_fun);

    s.C_t.col(s.time) = f.cwiseProduct(C_tm1) + i.cwiseProduct(ct);
    s.h_t.col(s.time) = o.cwiseProduct(s.C_t.col(s.time).unaryExpr(act_fun));
  }

  MatrixXd LSTMLayer::scan( const MatrixXd& x ) const {

    LSTMState state(x.cols(), m_n_outputs);

    for(state.time = 0; state.time < x.cols(); state.time++) {
      step( x.col( state.time ), state );
    }

    return state.h_t;
  }


  // GRU layer
  GRULayer::GRULayer(Activation activation, Activation inner_activation,
           MatrixXd W_z, MatrixXd U_z, VectorXd b_z,
           MatrixXd W_r, MatrixXd U_r, VectorXd b_r,
           MatrixXd W_h, MatrixXd U_h, VectorXd b_h):
    m_W_z(W_z),
    m_U_z(U_z),
    m_b_z(b_z),
    m_W_r(W_r),
    m_U_r(U_r),
    m_b_r(b_r),
    m_W_h(W_h),
    m_U_h(U_h),
    m_b_h(b_h)
  {
    m_n_outputs = m_W_h.rows();

    m_activation_fun = get_activation(activation);
    m_inner_activation_fun = get_activation(inner_activation);
  }
  // internal structure created on each scan call
  struct GRUState {
    GRUState(size_t n_input, size_t n_outputs);
    MatrixXd h_t;
    int time;
  };
  GRUState::GRUState(size_t n_input, size_t n_output):
    h_t(MatrixXd::Zero(n_output, n_input)),
    time(0)
  {
  }

  void GRULayer::step( const VectorXd& x_t, GRUState& s) const {
    // https://github.com/fchollet/keras/blob/master/keras/layers/recurrent.py#L547

    const auto& act_fun = m_activation_fun;
    const auto& in_act_fun = m_inner_activation_fun;

    int tm1 = s.time == 0 ? 0 : s.time - 1;
    VectorXd h_tm1 = s.h_t.col(tm1);
    VectorXd z  = (m_W_z*x_t + m_b_z + m_U_z*h_tm1).unaryExpr(in_act_fun);
    VectorXd r  = (m_W_r*x_t + m_b_r + m_U_r*h_tm1).unaryExpr(in_act_fun);
    VectorXd rh = r.cwiseProduct(h_tm1);
    VectorXd hh = (m_W_h*x_t + m_b_h + m_U_h*rh).unaryExpr(act_fun);
    VectorXd one = VectorXd::Ones(z.size());
    s.h_t.col(s.time)  = z.cwiseProduct(h_tm1) + (one - z).cwiseProduct(hh);
  }

  MatrixXd GRULayer::scan( const MatrixXd& x ) const {

    GRUState state(x.cols(), m_n_outputs);

    for(state.time = 0; state.time < x.cols(); state.time++) {
      step( x.col( state.time ), state );
    }

    return state.h_t;
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
    case Activation::ELU: return nn_elu;
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

  double nn_elu( double x ){
    /* ELU function : https://arxiv.org/pdf/1511.07289.pdf
       f(x)=(x>=0)*x + ( (x<0)*alpha*(exp(x)-1) )
    */
    double alpha(1.0); // need support from any alpha param
    double exp_term = alpha * (std::exp(x)-1);
    return x>=0 ? x : exp_term;
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

  void throw_if_not_normalization(const LayerConfig& layer) {
    if (layer.sublayers.size() > 0) {
      throw NNConfigurationException("sublayers in normalization layer");
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
