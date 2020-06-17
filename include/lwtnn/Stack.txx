#include "Stack.hh"
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
  template<typename T>
  StackT<T>::StackT() {
    m_layers.push_back(new DummyLayerT<T>);
    m_layers.push_back(new UnaryActivationLayerT<T>({ Activation::SIGMOID, 0.0 }));
    m_layers.push_back(new BiasLayerT<T>(std::vector<T>{1, 1, 1, 1}));
    
    MatrixX<T> mat(4, 4);
    mat <<
      0, 0, 0, 1,
      0, 0, 1, 0,
      0, 1, 0, 0,
      1, 0, 0, 0;
    m_layers.push_back(new MatrixLayerT<T>(mat));
    m_n_outputs = 4;
  }

  // construct from LayerConfig
  template<typename T>
  StackT<T>::StackT(size_t n_inputs, const std::vector<LayerConfig>& layers,
               size_t skip) {
    for (size_t nnn = skip; nnn < layers.size(); nnn++) {
      n_inputs = add_layers(n_inputs, layers.at(nnn));
    }
    // the final assigned n_inputs is the number of output nodes
    m_n_outputs = n_inputs;
  }

  template<typename T>
  StackT<T>::~StackT() {
    for (auto& layer: m_layers) {
      delete layer;
      layer = 0;
    }
  }
  
  template<typename T>
  VectorX<T> StackT<T>::compute(VectorX<T> in) const {
    for (const auto& layer: m_layers) {
      in = layer->compute(in);
    }
    return in;
  }
  
  template<typename T>
  size_t StackT<T>::n_outputs() const {
    return m_n_outputs;
  }


  // Private Stack methods to add various types of layers
  //
  // top level add_layers method. This delegates to the other methods
  // below
  
  
  template<typename T>
  size_t StackT<T>::add_layers(size_t n_inputs, const LayerConfig& layer) {
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

  template<typename T>
  size_t StackT<T>::add_dense_layers(size_t n_inputs, const LayerConfig& layer) {
    assert(layer.architecture == Architecture::DENSE);
    throw_if_not_dense(layer);

    size_t n_outputs = n_inputs;

    // add matrix layer
    if (layer.weights.size() > 0) {
      MatrixX<T> matrix = build_matrixT<T>(layer.weights, n_inputs);
      n_outputs = matrix.rows();
      m_layers.push_back(new MatrixLayerT<T>(matrix));
    };

    // add bias layer
    if (layer.bias.size() > 0) {
      if (n_outputs != layer.bias.size() ) {
        std::string problem = "tried to add a bias layer with " +
          std::to_string(layer.bias.size()) + " entries, previous layer"
          " had " + std::to_string(n_outputs) + " outputs";
        throw NNConfigurationException(problem);
      }
      m_layers.push_back(new BiasLayerT<T>(layer.bias));
    }

    // add activation layer
    if (layer.activation.function != Activation::LINEAR) {
      m_layers.push_back(get_raw_activation_layerT<T>(layer.activation));
    }

    return n_outputs;
  }

  template<typename T>
  size_t StackT<T>::add_normalization_layers(size_t n_inputs, const LayerConfig& layer) {
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
    VectorX<T> v_weights = build_vector(layer.weights);
    VectorX<T> v_bias = build_vector(layer.bias);

    m_layers.push_back(
      new NormalizationLayerT<T>(v_weights, v_bias));
    return n_inputs;
  }


  template<typename T>
  size_t StackT<T>::add_highway_layers(size_t n_inputs, const LayerConfig& layer) {
    auto& comps = layer.components;
    const auto& t = get_component(comps.at(Component::T), n_inputs);
    const auto& c = get_component(comps.at(Component::CARRY), n_inputs);

    m_layers.push_back(
      new HighwayLayerT<T>(t.W, t.b, c.W, c.b, layer.activation));
    return n_inputs;
  }


  template<typename T>
  size_t StackT<T>::add_maxout_layers(size_t n_inputs, const LayerConfig& layer) {
    assert(layer.architecture == Architecture::MAXOUT);
    throw_if_not_maxout(layer);
    std::vector<typename MaxoutLayerT<T>::InitUnit> matrices;
    std::set<size_t> n_outputs;
    for (const auto& sublayer: layer.sublayers) {
      MatrixX<T> matrix = build_matrix(sublayer.weights, n_inputs);
      VectorX<T> bias = build_vector(sublayer.bias);
      n_outputs.insert(matrix.rows());
      matrices.push_back(std::make_pair(matrix, bias));
    }
    if (n_outputs.size() == 0) {
      throw NNConfigurationException("tried to build maxout withoutweights!");
    }
    else if (n_outputs.size() != 1) {
      throw NNConfigurationException("uneven matrices for maxout");
    }
    m_layers.push_back(new MaxoutLayerT<T>(matrices));
    return *n_outputs.begin();
  }


  // _______________________________________________________________________
  // Feed-forward layers

  template<typename T>
  VectorX<T> DummyLayerT<T>::compute(const VectorX<T>& in) const {
    return in;
  }

  // activation functions
  template<typename T>
  UnaryActivationLayerT<T>::UnaryActivationLayerT(ActivationConfig act):
    m_func(get_activationT<T>(act))
  {
  }
  
  template<typename T>
  VectorX<T> UnaryActivationLayerT<T>::compute(const VectorX<T>& in) const {
    return in.unaryExpr(m_func);
  }

  template<typename T>
  VectorX<T> SoftmaxLayerT<T>::compute(const VectorX<T>& in) const {
    // More numerically stable softmax, as suggested in
    // http://stackoverflow.com/a/34969389
    size_t n_elements = in.rows();
    VectorX<T> expv(n_elements);
    T max = in.maxCoeff();
    for (size_t iii = 0; iii < n_elements; iii++) {
      using std::exp; // weird autodiff ...
      expv(iii) = exp(in(iii) - max);
    }
    T sum_exp = expv.sum();
    return expv / sum_exp;
  }

  // bias layer
  template<typename T>
  BiasLayerT<T>::BiasLayerT(const VectorX<T>& bias): m_bias(bias)
  {
  }
  
  template<typename T> 
  template<typename U>
  BiasLayerT<T>::BiasLayerT(const std::vector<U>& bias):
    m_bias(build_vectorT<T,U>(bias))
  {
  }
  
  template<typename T>
  VectorX<T> BiasLayerT<T>::compute(const VectorX<T>& in) const {
    return in + m_bias;
  }

  // basic dense matrix layer
  template<typename T>
  MatrixLayerT<T>::MatrixLayerT(const MatrixX<T>& matrix):
    m_matrix(matrix)
  {
  }
  
  template<typename T>
  VectorX<T> MatrixLayerT<T>::compute(const VectorX<T>& in) const {
    return m_matrix * in;
  }

  // maxout layer
  template<typename T>
  MaxoutLayerT<T>::MaxoutLayerT(const std::vector<MaxoutLayerT::InitUnit>& units):
    m_bias(units.size(), units.front().first.rows())
  {
    int out_pos = 0;
    for (const auto& unit: units) {
      m_matrices.push_back(unit.first);
      m_bias.row(out_pos) = unit.second;
      out_pos++;
    }
  }
  
  template<typename T>
  VectorX<T> MaxoutLayerT<T>::compute(const VectorX<T>& in) const {
    // eigen supports tensors, but only in the experimental component
    // for now just stick to matrix and vector classes
    const size_t n_mat = m_matrices.size();
    const size_t out_dim = m_matrices.front().rows();
    MatrixX<T> outputs(n_mat, out_dim);
    for (size_t mat_n = 0; mat_n < n_mat; mat_n++) {
      outputs.row(mat_n) = m_matrices.at(mat_n) * in;
    }
    outputs += m_bias;
    return outputs.colwise().maxCoeff();
  }

   // Normalization layer
  template<typename T>
   NormalizationLayerT<T>::NormalizationLayerT(const VectorX<T>& W,
                                          const VectorX<T>& b):
    _W(W), _b(b)
  {
  }
  
  
  template<typename T>
  VectorX<T> NormalizationLayerT<T>::compute(const VectorX<T>& in) const {
    VectorX<T> shift = in + _b ;
    return _W.cwiseProduct(shift);
  }

  // highway layer
  template<typename T>
  HighwayLayerT<T>::HighwayLayerT(const MatrixX<T>& W,
                             const VectorX<T>& b,
                             const MatrixX<T>& W_carry,
                             const VectorX<T>& b_carry,
                             ActivationConfig activation):
    m_w_t(W), m_b_t(b), m_w_c(W_carry), m_b_c(b_carry),
    m_act(get_activationT<T>(activation))
  {
  }
  
  
  template<typename T>
  VectorX<T> HighwayLayerT<T>::compute(const VectorX<T>& in) const {
    const std::function<T(T)> sig(nn_sigmoidT<T>);
    ArrayX<T> c = (m_w_c * in + m_b_c).unaryExpr(sig);
    ArrayX<T> t = (m_w_t * in + m_b_t).unaryExpr(m_act);
    return c * t + (1 - c) * in.array();
  }

  // ______________________________________________________________________
  // Recurrent Stack

  template<typename T>
  RecurrentStackT<T>::RecurrentStackT(size_t n_inputs,
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
  
  template<typename T>
  RecurrentStackT<T>::~RecurrentStackT() {
    for (auto& layer: m_layers) {
      delete layer;
      layer = 0;
    }
  }
  
  template<typename T>
  MatrixX<T> RecurrentStackT<T>::scan(MatrixX<T> in) const {
    for (auto* layer: m_layers) {
      in = layer->scan(in);
    }
    return in;
  }
  
  template<typename T>
  size_t RecurrentStackT<T>::n_outputs() const {
    return m_n_outputs;
  }

  template<typename T>
  size_t RecurrentStackT<T>::add_lstm_layers(size_t n_inputs,
                                         const LayerConfig& layer) {
    auto& comps = layer.components;
    const auto& i = get_component(comps.at(Component::I), n_inputs);
    const auto& o = get_component(comps.at(Component::O), n_inputs);
    const auto& f = get_component(comps.at(Component::F), n_inputs);
    const auto& c = get_component(comps.at(Component::C), n_inputs);
    m_layers.push_back(
      new LSTMLayerT<T>(layer.activation, layer.inner_activation,
                    i.W, i.U, i.b,
                    f.W, f.U, f.b,
                    o.W, o.U, o.b,
                    c.W, c.U, c.b));
    return o.b.rows();
  }

  template<typename T>
  size_t RecurrentStackT<T>::add_gru_layers(size_t n_inputs,
                                         const LayerConfig& layer) {
    auto& comps = layer.components;
    const auto& z = get_component(comps.at(Component::Z), n_inputs);
    const auto& r = get_component(comps.at(Component::R), n_inputs);
    const auto& h = get_component(comps.at(Component::H), n_inputs);
    m_layers.push_back(
      new GRULayerT<T>(layer.activation, layer.inner_activation,
                    z.W, z.U, z.b,
                    r.W, r.U, r.b,
                    h.W, h.U, h.b));
    return h.b.rows();
  }

  template<typename T>
  size_t RecurrentStackT<T>::add_embedding_layers(size_t n_inputs,
                                              const LayerConfig& layer) {
    for (const auto& emb: layer.embedding) {
      size_t n_wt = emb.weights.size();
      size_t n_cats = n_wt / emb.n_out;
      MatrixX<T> mat = build_matrix(emb.weights, n_cats);
      m_layers.push_back(new EmbeddingLayerT<T>(emb.index, mat));
      n_inputs += emb.n_out - 1;
    }
    return n_inputs;
  }

  template<typename T>
  ReductionStackT<T>::ReductionStackT(size_t n_in,
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
  
  template<typename T>
  ReductionStackT<T>::~ReductionStackT() {
    delete m_recurrent;
    delete m_stack;
  }
  
  template<typename T>
  VectorX<T> ReductionStackT<T>::reduce(MatrixX<T> in) const {
    in = m_recurrent->scan(in);
    return m_stack->compute(in.col(in.cols() -1));
  }
  
  template<typename T>
  size_t ReductionStackT<T>::n_outputs() const {
    return m_stack->n_outputs();
  }

  // __________________________________________________________________
  // Recurrent layers

  template<typename T>
  EmbeddingLayerT<T>::EmbeddingLayerT(int var_row_index, MatrixX<T> W):
    m_var_row_index(var_row_index),
    m_W(W)
  {
    if(var_row_index < 0)
      throw NNConfigurationException(
        "EmbeddingLayer::EmbeddingLayer - can not set var_row_index<0,"
        " it is an index for a matrix row!");
  }

  template<typename T>
  MatrixX<T> EmbeddingLayerT<T>::scan( const MatrixX<T>& x) const {

    if( m_var_row_index >= x.rows() )
      throw NNEvaluationException(
        "EmbeddingLayer::scan - var_row_index is larger than input matrix"
        " number of rows!");

    MatrixX<T> embedded(m_W.rows(), x.cols());

    for(int icol=0; icol<x.cols(); icol++) {
      using std::floor; // weird autodiff
      T vector_idx = x(m_var_row_index, icol);
      bool is_int = floor(vector_idx) == vector_idx;
      bool is_valid = (vector_idx >= 0) && (vector_idx < m_W.cols());
      if (!is_int || !is_valid) throw NNEvaluationException(
        "Invalid embedded index: " + std::to_string(vector_idx));
      embedded.col(icol) = m_W.col( vector_idx );
    }

    //only embed 1 variable at a time, so this should be correct size
    MatrixX<T> out(m_W.rows() + (x.rows() - 1), x.cols());

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
  template<typename T>
  LSTMLayerT<T>::LSTMLayerT(ActivationConfig activation,
                       ActivationConfig inner_activation,
                       MatrixX<T> W_i, MatrixX<T> U_i, VectorX<T> b_i,
                       MatrixX<T> W_f, MatrixX<T> U_f, VectorX<T> b_f,
                       MatrixX<T> W_o, MatrixX<T> U_o, VectorX<T> b_o,
                       MatrixX<T> W_c, MatrixX<T> U_c, VectorX<T> b_c):
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
  template<typename T>
  struct LSTMStateT {
    LSTMStateT(size_t n_input, size_t n_outputs);
    MatrixX<T> C_t;
    MatrixX<T> h_t;
    int time;
  };
  
  template<typename T>
  LSTMStateT<T>::LSTMStateT(size_t n_input, size_t n_output):
    C_t(MatrixX<T>::Zero(n_output, n_input)),
    h_t(MatrixX<T>::Zero(n_output, n_input)),
    time(0)
  {
  }

  template<typename T>
  void LSTMLayerT<T>::step(const VectorX<T>& x_t, LSTMState& s) const {
    // https://github.com/fchollet/keras/blob/master/keras/layers/recurrent.py#L740

    const auto& act_fun = m_activation_fun;
    const auto& in_act_fun = m_inner_activation_fun;

    int tm1 = s.time == 0 ? 0 : s.time - 1;
    VectorX<T> h_tm1 = s.h_t.col(tm1);
    VectorX<T> C_tm1 = s.C_t.col(tm1);

    VectorX<T> i  =  (m_W_i*x_t + m_b_i + m_U_i*h_tm1).unaryExpr(in_act_fun);
    VectorX<T> f  =  (m_W_f*x_t + m_b_f + m_U_f*h_tm1).unaryExpr(in_act_fun);
    VectorX<T> o  =  (m_W_o*x_t + m_b_o + m_U_o*h_tm1).unaryExpr(in_act_fun);
    VectorX<T> ct =  (m_W_c*x_t + m_b_c + m_U_c*h_tm1).unaryExpr(act_fun);

    s.C_t.col(s.time) = f.cwiseProduct(C_tm1) + i.cwiseProduct(ct);
    s.h_t.col(s.time) = o.cwiseProduct(s.C_t.col(s.time).unaryExpr(act_fun));
  }

  template<typename T>
  MatrixX<T> LSTMLayerT<T>::scan( const MatrixX<T>& x ) const {

    LSTMStateT<T> state(x.cols(), m_n_outputs);

    for(state.time = 0; state.time < x.cols(); state.time++) {
      step( x.col( state.time ), state );
    }

    return state.h_t;
  }


  // GRU layer
  template<typename T>
  GRULayerT<T>::GRULayerT(ActivationConfig activation,
                     ActivationConfig inner_activation,
                     MatrixX<T> W_z, MatrixX<T> U_z, VectorX<T> b_z,
                     MatrixX<T> W_r, MatrixX<T> U_r, VectorX<T> b_r,
                     MatrixX<T> W_h, MatrixX<T> U_h, VectorX<T> b_h):
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
  template<typename T>
  struct GRUStateT {
    GRUStateT(size_t n_input, size_t n_outputs);
    MatrixX<T> h_t;
    int time;
  };
  
  template<typename T>
  GRUStateT<T>::GRUStateT(size_t n_input, size_t n_output):
    h_t(MatrixX<T>::Zero(n_output, n_input)),
    time(0)
  {
  }

  template<typename T>
  void GRULayerT<T>::step( const VectorX<T>& x_t, GRUState& s) const {
    // https://github.com/fchollet/keras/blob/master/keras/layers/recurrent.py#L547

    const auto& act_fun = m_activation_fun;
    const auto& in_act_fun = m_inner_activation_fun;

    int tm1 = s.time == 0 ? 0 : s.time - 1;
    VectorX<T> h_tm1 = s.h_t.col(tm1);
    VectorX<T> z  = (m_W_z*x_t + m_b_z + m_U_z*h_tm1).unaryExpr(in_act_fun);
    VectorX<T> r  = (m_W_r*x_t + m_b_r + m_U_r*h_tm1).unaryExpr(in_act_fun);
    VectorX<T> rh = r.cwiseProduct(h_tm1);
    VectorX<T> hh = (m_W_h*x_t + m_b_h + m_U_h*rh).unaryExpr(act_fun);
    VectorX<T> one = VectorX<T>::Ones(z.size());
    s.h_t.col(s.time)  = z.cwiseProduct(h_tm1) + (one - z).cwiseProduct(hh);
  }

  template<typename T>
  MatrixX<T> GRULayerT<T>::scan( const MatrixX<T>& x ) const {

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
  
  
  template<typename T>
  ILayerT<T>* get_raw_activation_layerT(ActivationConfig activation) {
    // Check for special cases. If it's not one, use
    // UnaryActivationLayer
    switch (activation.function) {
    case Activation::SOFTMAX: return new SoftmaxLayerT<T>;
    default: return new UnaryActivationLayerT<T>(activation);
    }
  }

  // Most activation functions should be handled here.
  template<typename T>
  std::function<T(T)> get_activationT(lwt::ActivationConfig act) {
    using namespace lwt;
    switch (act.function) {
    case Activation::SIGMOID: return nn_sigmoidT<T>;
    case Activation::HARD_SIGMOID: return nn_hard_sigmoidT<T>;
    case Activation::SWISH: return SwishT<T>(act.alpha);
    case Activation::TANH: return nn_tanhT<T>;
    case Activation::RECTIFIED: return nn_reluT<T>;
    case Activation::ELU: return ELUT<T>(act.alpha);
    case Activation::LEAKY_RELU: return LeakyReLUT<T>(act.alpha);
    case Activation::LINEAR: return [](T x){return x;};
    case Activation::ABS: return [](T x){using std::abs; /* autodiff... */ return abs(x);};
    default: {
      throw NNConfigurationException("Got undefined activation function");
    }
    }
  }

  template<typename T>
  T nn_sigmoidT( T x ){
    //github.com/Theano/Theano/blob/master/theano/tensor/nnet/sigm.py#L35
    using std::exp;  // strange requirement from autodiff
    if (x < -30.0) return 0.0;
    if (x >  30.0) return 1.0;
    return 1.0 / (1.0 + exp(-1.0*x));
  }

  template<typename T>
  T nn_hard_sigmoidT( T x ){
    //github.com/Theano/Theano/blob/master/theano/tensor/nnet/sigm.py#L279
    T out = 0.2*x + 0.5;
    if (out < 0) return 0.0;
    if (out > 1) return 1.0;
    return out;
  }

  template<typename T>
  SwishT<T>::SwishT(T alpha):
    m_alpha(alpha)
  {}
  
  template<typename T>
  T SwishT<T>::operator()(T x) const {
    return x * nn_sigmoidT<T>(m_alpha * x);
  }

  template<typename T>
  T nn_tanhT( T x ){
    using std::tanh; // strange requirement from autodiff
    return tanh(x);
  }

  template<typename T>
  T nn_reluT( T x) {
    if (std::isnan(static_cast<double>(x))) return x;
    else return x > 0 ? x : 0;
  }

  template<typename T>
  ELUT<T>::ELUT(T alpha):
    m_alpha(alpha)
  {}
  
  template<typename T>
  T ELUT<T>::operator()( T x ) const {
    /* ELU function : https://arxiv.org/pdf/1511.07289.pdf
       f(x)=(x>=0)*x + ( (x<0)*alpha*(exp(x)-1) )
    */
    using std::exp; // strange requirement from autodiff
    T exp_term = m_alpha * (exp(x)-1);
    return x>=0 ? x : exp_term;
  }

  template<typename T>
  LeakyReLUT<T>::LeakyReLUT(T alpha):
    m_alpha(alpha)
  {}
  
  template<typename T>
  T LeakyReLUT<T>::operator()(T x) const {
    return x > 0 ? x : static_cast<T>(m_alpha * x); // weird autodiff
  }

  // ________________________________________________________________________
  // utility functions
  template<typename T1, typename T2>
  MatrixX<T1> build_matrixT(const std::vector<T2>& weights, size_t n_inputs)
  {
    static_assert( std::is_same<T1, T2>::value ||
                   std::is_assignable<T1, T2>::value, 
                   "T2 cannot be implicitly assigned to T1" );
    
    size_t n_elements = weights.size();
    if ((n_elements % n_inputs) != 0) {
      std::string problem = "matrix elements not divisible by number"
        " of columns. Elements: " + std::to_string(n_elements) +
        ", Inputs: " + std::to_string(n_inputs);
      throw lwt::NNConfigurationException(problem);
    }
    size_t n_outputs = n_elements / n_inputs;
    MatrixX<T1> matrix(n_outputs, n_inputs);
    for (size_t row = 0; row < n_outputs; row++) {
      for (size_t col = 0; col < n_inputs; col++) {
        T1 element = weights.at(col + row * n_inputs);
        matrix(row, col) = element;
      }
    }
    return matrix;
  }
  
  template<typename T1, typename T2>
  VectorX<T1> build_vectorT(const std::vector<T2>& bias) 
  {
    static_assert( std::is_same<T1, T2>::value ||
                   std::is_assignable<T1, T2>::value, 
                   "T2 cannot be implicitly assigned to T1" );
    
    VectorX<T1> out(bias.size());
    size_t idx = 0;
    for (const auto& val: bias) {
      out(idx) = val;
      idx++;
    }
    return out;
  }

  // component-wise getters (for Highway, lstm, etc)
  template<typename T>
  DenseComponentsT<T> get_componentT(const lwt::LayerConfig& layer, size_t n_in) {
    using namespace Eigen;
    using namespace lwt;
    MatrixX<T> weights = build_matrixT<T, double>(layer.weights, n_in);
    size_t n_out = weights.rows();
    VectorX<T> bias = build_vectorT<T, double>(layer.bias);

    // the u element is optional
    size_t u_el = layer.U.size();
    MatrixX<T> U = u_el ? build_matrixT<T, double>(layer.U, n_out) : MatrixX<T>::Zero(0,0);

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
