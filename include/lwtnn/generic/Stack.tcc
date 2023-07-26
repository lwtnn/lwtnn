#ifndef LWTNN_GENERIC_STACK_TCC
#define LWTNN_GENERIC_STACK_TCC

#include "lwtnn/generic/Stack.hh"

#include <set>


namespace lwt {
namespace generic {

  // ______________________________________________________________________
  // Feed forward Stack class

  // dummy construction routine
  template<typename T>
  Stack<T>::Stack() {
    m_layers.push_back(new DummyLayer<T>);
    m_layers.push_back(new UnaryActivationLayer<T>({ Activation::SIGMOID, 0.0 }));
    m_layers.push_back(new BiasLayer<T>(std::vector<T>{1, 1, 1, 1}));

    MatrixX<T> mat(4, 4);
    mat <<
      0, 0, 0, 1,
      0, 0, 1, 0,
      0, 1, 0, 0,
      1, 0, 0, 0;
    m_layers.push_back(new MatrixLayer<T>(mat));
    m_n_outputs = 4;
  }

  // construct from LayerConfig
  template<typename T>
  Stack<T>::Stack(std::size_t n_inputs, const std::vector<LayerConfig>& layers,
               std::size_t skip) {
    for (std::size_t nnn = skip; nnn < layers.size(); nnn++) {
      n_inputs = add_layers(n_inputs, layers.at(nnn));
    }
    // the final assigned n_inputs is the number of output nodes
    m_n_outputs = n_inputs;
  }

  template<typename T>
  Stack<T>::~Stack() {
    for (auto& layer: m_layers) {
      delete layer;
      layer = 0;
    }
  }

  template<typename T>
  VectorX<T> Stack<T>::compute(VectorX<T> in) const {
    for (const auto& layer: m_layers) {
      in = layer->compute(in);
    }
    return in;
  }

  template<typename T>
  std::size_t Stack<T>::n_outputs() const {
    return m_n_outputs;
  }


  // Private Stack methods to add various types of layers
  //
  // top level add_layers method. This delegates to the other methods
  // below


  template<typename T>
  std::size_t Stack<T>::add_layers(std::size_t n_inputs, const LayerConfig& layer) {
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
  std::size_t Stack<T>::add_dense_layers(std::size_t n_inputs, const LayerConfig& layer) {
    assert(layer.architecture == Architecture::DENSE);
    throw_if_not_dense(layer);

    std::size_t n_outputs = n_inputs;

    // add matrix layer
    if (layer.weights.size() > 0) {
      MatrixX<T> matrix = build_matrix<T>(layer.weights, n_inputs);
      n_outputs = matrix.rows();
      m_layers.push_back(new MatrixLayer<T>(matrix));
    };

    // add bias layer
    if (layer.bias.size() > 0) {
      if (n_outputs != layer.bias.size() ) {
        std::string problem = "tried to add a bias layer with " +
          std::to_string(layer.bias.size()) + " entries, previous layer"
          " had " + std::to_string(n_outputs) + " outputs";
        throw NNConfigurationException(problem);
      }
      m_layers.push_back(new BiasLayer<T>(layer.bias));
    }

    // add activation layer
    if (layer.activation.function != Activation::LINEAR) {
      m_layers.push_back(get_raw_activation_layer<T>(layer.activation));
    }

    return n_outputs;
  }

  template<typename T>
  std::size_t Stack<T>::add_normalization_layers(std::size_t n_inputs, const LayerConfig& layer) {
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
    VectorX<T> v_weights = build_vector<T>(layer.weights);
    VectorX<T> v_bias = build_vector<T>(layer.bias);

    m_layers.push_back(
      new NormalizationLayer<T>(v_weights, v_bias));
    return n_inputs;
  }


  template<typename T>
  std::size_t Stack<T>::add_highway_layers(std::size_t n_inputs, const LayerConfig& layer) {
    auto& comps = layer.components;
    const auto& t = get_component<T>(comps.at(Component::T), n_inputs);
    const auto& c = get_component<T>(comps.at(Component::CARRY), n_inputs);

    m_layers.push_back(
      new HighwayLayer<T>(t.W, t.b, c.W, c.b, layer.activation));
    return n_inputs;
  }


  template<typename T>
  std::size_t Stack<T>::add_maxout_layers(std::size_t n_inputs, const LayerConfig& layer) {
    assert(layer.architecture == Architecture::MAXOUT);
    throw_if_not_maxout(layer);
    std::vector<typename MaxoutLayer<T>::InitUnit> matrices;
    std::set<std::size_t> n_outputs;
    for (const auto& sublayer: layer.sublayers) {
      MatrixX<T> matrix = build_matrix<T>(sublayer.weights, n_inputs);
      VectorX<T> bias = build_vector<T>(sublayer.bias);
      n_outputs.insert(matrix.rows());
      matrices.push_back(std::make_pair(matrix, bias));
    }
    if (n_outputs.size() == 0) {
      throw NNConfigurationException("tried to build maxout withoutweights!");
    }
    else if (n_outputs.size() != 1) {
      throw NNConfigurationException("uneven matrices for maxout");
    }
    m_layers.push_back(new MaxoutLayer<T>(matrices));
    return *n_outputs.begin();
  }


  // _______________________________________________________________________
  // Feed-forward layers

  template<typename T>
  VectorX<T> DummyLayer<T>::compute(const VectorX<T>& in) const {
    return in;
  }

  // activation functions
  template<typename T>
  UnaryActivationLayer<T>::UnaryActivationLayer(ActivationConfig act):
    m_func(get_activation<T>(act))
  {
  }

  template<typename T>
  VectorX<T> UnaryActivationLayer<T>::compute(const VectorX<T>& in) const {
    return in.unaryExpr(m_func);
  }

  template<typename T>
  VectorX<T> SoftmaxLayer<T>::compute(const VectorX<T>& in) const {
    // More numerically stable softmax, as suggested in
    // http://stackoverflow.com/a/34969389
    std::size_t n_elements = in.rows();
    VectorX<T> expv(n_elements);
    T max = in.maxCoeff();
    for (std::size_t iii = 0; iii < n_elements; iii++) {
      using std::exp; // weird autodiff ...
      expv(iii) = exp(in(iii) - max);
    }
    T sum_exp = expv.sum();
    return expv / sum_exp;
  }

  template<typename T>
  VectorX<T> ReLULayer<T>::compute(const VectorX<T>& in) const {
    return in.unaryExpr([](T x){ return x < 0 ? 0 : x; });
  }

  template<typename T>
  VectorX<T> SigmoidLayer<T>::compute(const VectorX<T>& in) const {
    using std::exp;  // strange requirement from autodiff
    return in.unaryExpr([](T x) -> T {
      // These checks are to prevent a floating point overflow or
      // underflow in the exponentiation. Neither under or overflows
      // are a problem for the output (an overflow in the exp should
      // still return 0, and an underflow should return 1), but ATLAS
      // runs FPE auditors and the overflows make them sad.
      if (x < -30.0) return 0.0;
      if (x >  30.0) return 1.0;
      return 1.0 / (1.0 + exp(-1.0*x));
    });
  }

  template<typename T>
  VectorX<T> HardSigmoidLayer<T>::compute(const VectorX<T>& in) const {
    //https://github.com/Theano/Theano/blob/eb6a4125c4f5617e74b10503afc3f334f17cf545/theano/tensor/nnet/sigm.py#L279
    return in.unaryExpr([](T x) -> T {
      T out = 0.2*x + 0.5;
      if (out < 0) return 0.0;
      if (out > 1) return 1.0;
      return out;
    });
  }

  template<typename T>
  VectorX<T> TanhLayer<T>::compute(const VectorX<T>& in) const {
    return tanh(in.array());
  }

  template<typename T>
  VectorX<T> AbsLayer<T>::compute(const VectorX<T>& in) const {
    return abs(in.array());
  }

  // bias layer
  template<typename T>
  BiasLayer<T>::BiasLayer(const VectorX<T>& bias): m_bias(bias)
  {
  }

  template<typename T>
  template<typename U>
  BiasLayer<T>::BiasLayer(const std::vector<U>& bias):
    m_bias(build_vector<T,U>(bias))
  {
  }

  template<typename T>
  VectorX<T> BiasLayer<T>::compute(const VectorX<T>& in) const {
    return in + m_bias;
  }

  // basic dense matrix layer
  template<typename T>
  MatrixLayer<T>::MatrixLayer(const MatrixX<T>& matrix):
    m_matrix(matrix)
  {
  }

  template<typename T>
  VectorX<T> MatrixLayer<T>::compute(const VectorX<T>& in) const {
    return m_matrix * in;
  }

  // maxout layer
  template<typename T>
  MaxoutLayer<T>::MaxoutLayer(const std::vector<MaxoutLayer::InitUnit>& units):
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
  VectorX<T> MaxoutLayer<T>::compute(const VectorX<T>& in) const {
    // eigen supports tensors, but only in the experimental component
    // for now just stick to matrix and vector classes
    const std::size_t n_mat = m_matrices.size();
    const std::size_t out_dim = m_matrices.front().rows();
    MatrixX<T> outputs(n_mat, out_dim);
    for (std::size_t mat_n = 0; mat_n < n_mat; mat_n++) {
      outputs.row(mat_n) = m_matrices.at(mat_n) * in;
    }
    outputs += m_bias;
    return outputs.colwise().maxCoeff();
  }

   // Normalization layer
  template<typename T>
   NormalizationLayer<T>::NormalizationLayer(const VectorX<T>& W,
                                          const VectorX<T>& b):
    _W(W), _b(b)
  {
  }


  template<typename T>
  VectorX<T> NormalizationLayer<T>::compute(const VectorX<T>& in) const {
    VectorX<T> shift = in + _b ;
    return _W.cwiseProduct(shift);
  }

  // highway layer
  template<typename T>
  HighwayLayer<T>::HighwayLayer(const MatrixX<T>& W,
                             const VectorX<T>& b,
                             const MatrixX<T>& W_carry,
                             const VectorX<T>& b_carry,
                             ActivationConfig activation):
    m_w_t(W), m_b_t(b), m_w_c(W_carry), m_b_c(b_carry),
    m_act(get_raw_activation_layer<T>(activation))
  {
  }


  template<typename T>
  VectorX<T> HighwayLayer<T>::compute(const VectorX<T>& in) const {
    const std::function<T(T)> sig(nn_sigmoid<T>);
    ArrayX<T> c = (m_w_c * in + m_b_c).unaryExpr(sig);
    ArrayX<T> t = m_act->compute(m_w_t * in + m_b_t);
    return c * t + (1 - c) * in.array();
  }

  // ______________________________________________________________________
  // Recurrent Stack

  template<typename T>
  RecurrentStack<T>::RecurrentStack(std::size_t n_inputs,
                                 const std::vector<lwt::LayerConfig>& layers)
  {
    using namespace lwt;
    const std::size_t n_layers = layers.size();
    for (std::size_t layer_n = 0; layer_n < n_layers; layer_n++) {
      auto& layer = layers.at(layer_n);

      // add recurrent layers (now LSTM and GRU!)
      if (layer.architecture == Architecture::LSTM) {
        n_inputs = add_lstm_layers(n_inputs, layer);
      } else if (layer.architecture == Architecture::GRU) {
        n_inputs = add_gru_layers(n_inputs, layer);
      } else if (layer.architecture == Architecture::SIMPLERNN) {
        n_inputs = add_simplernn_layers(n_inputs, layer);
      } else if (layer.architecture == Architecture::CONV1D){
        n_inputs = add_conv1d_layers(n_inputs, layer);
      } else if (layer.architecture == Architecture::EMBEDDING) {
        n_inputs = add_embedding_layers(n_inputs, layer);
      } else {
        throw NNConfigurationException("found non-recurrent layer");
      }
    }
    m_n_outputs = n_inputs;
  }

  template<typename T>
  RecurrentStack<T>::~RecurrentStack() {
    for (auto& layer: m_layers) {
      delete layer;
      layer = 0;
    }
  }

  template<typename T>
  MatrixX<T> RecurrentStack<T>::scan(MatrixX<T> in) const {
    for (auto* layer: m_layers) {
      in = layer->scan(in);
    }
    return in;
  }

  template<typename T>
  std::size_t RecurrentStack<T>::n_outputs() const {
    return m_n_outputs;
  }

  template<typename T>
  std::size_t RecurrentStack<T>::add_lstm_layers(std::size_t n_inputs,
                                         const LayerConfig& layer) {
    auto& comps = layer.components;
    const auto& i = get_component<T>(comps.at(Component::I), n_inputs);
    const auto& o = get_component<T>(comps.at(Component::O), n_inputs);
    const auto& f = get_component<T>(comps.at(Component::F), n_inputs);
    const auto& c = get_component<T>(comps.at(Component::C), n_inputs);
    m_layers.push_back(
      new LSTMLayer<T>(layer.activation, layer.inner_activation,
                    i.W, i.U, i.b,
                    f.W, f.U, f.b,
                    o.W, o.U, o.b,
                    c.W, c.U, c.b));
    return o.b.rows();
  }

  template<typename T>
  std::size_t RecurrentStack<T>::add_gru_layers(std::size_t n_inputs,
                                         const LayerConfig& layer) {
    auto& comps = layer.components;
    const auto& z = get_component<T>(comps.at(Component::Z), n_inputs);
    const auto& r = get_component<T>(comps.at(Component::R), n_inputs);
    const auto& h = get_component<T>(comps.at(Component::H), n_inputs);
    m_layers.push_back(
      new GRULayer<T>(layer.activation, layer.inner_activation,
                    z.W, z.U, z.b,
                    r.W, r.U, r.b,
                    h.W, h.U, h.b));
    return h.b.rows();
  }

  template<typename T>
  std::size_t RecurrentStack<T>::add_simplernn_layers(std::size_t n_inputs,
                                         const LayerConfig& layer) {
    auto& comps = layer.components;
    const auto& h = get_component<T>(comps.at(Component::H), n_inputs);
    m_layers.push_back(
      new SimpleRNNLayer<T>(layer.activation, h.W, h.U, h.b));
    return h.b.rows();
  }

  template<typename T>
  std::size_t RecurrentStack<T>::add_conv1d_layers(std::size_t n_inputs, const LayerConfig& layer) {
    assert(layer.architecture == Architecture::CONV1D);
    throw_if_not_conv1d(layer);

    std::size_t n_outputs = n_inputs;

    if (layer.weights.size() > 0) {
      std::size_t kernel_size = layer.weights.size()/n_inputs/layer.bias.size();

      VectorX<T> bias = build_vector<T>(layer.bias);
      n_outputs = layer.bias.size();

      // Construct weights matrix with weights operating on one input node per row
      MatrixX<T> matrix = MatrixX<T>::Zero(kernel_size*n_outputs, n_inputs);
      for(std::size_t k=0; k<kernel_size; k++){
        for(std::size_t o=0; o<n_outputs; o++){
          for(std::size_t i=0; i<n_inputs; i++){
            T element = layer.weights.at(o*n_inputs*kernel_size + i*kernel_size + k);
            matrix(o*kernel_size + k, i) = element;
          }
        }
      }
      m_layers.push_back(new Conv1dLayer<T>(layer.activation, matrix, bias, layer.conv1d));
    }

    return n_outputs;
  }

  template<typename T>
  std::size_t RecurrentStack<T>::add_embedding_layers(std::size_t n_inputs,
                                              const LayerConfig& layer) {
    for (const auto& emb: layer.embedding) {
      std::size_t n_wt = emb.weights.size();
      std::size_t n_cats = n_wt / emb.n_out;
      MatrixX<T> mat = build_matrix<T>(emb.weights, n_cats);
      m_layers.push_back(new EmbeddingLayer<T>(emb.index, mat));
      n_inputs += emb.n_out - 1;
    }
    return n_inputs;
  }

  template<typename T>
  ReductionStack<T>::ReductionStack(std::size_t n_in,
                                 const std::vector<LayerConfig>& layers) {
    std::vector<LayerConfig> recurrent;
    std::vector<LayerConfig> feed_forward;
    std::set<Architecture> recurrent_arcs{
      Architecture::LSTM, Architecture::GRU, Architecture::SIMPLERNN, Architecture::EMBEDDING};
    for (const auto& layer: layers) {
      if (recurrent_arcs.count(layer.architecture)) {
        recurrent.push_back(layer);
      } else {
        feed_forward.push_back(layer);
      }
    }
    m_recurrent = new RecurrentStack<T>(n_in, recurrent);
    m_stack = new Stack<T>(m_recurrent->n_outputs(), feed_forward);
  }

  template<typename T>
  ReductionStack<T>::~ReductionStack() {
    delete m_recurrent;
    delete m_stack;
  }

  template<typename T>
  VectorX<T> ReductionStack<T>::reduce(MatrixX<T> in) const {
    in = m_recurrent->scan(in);
    return m_stack->compute(in.col(in.cols() -1));
  }

  template<typename T>
  std::size_t ReductionStack<T>::n_outputs() const {
    return m_stack->n_outputs();
  }

  // __________________________________________________________________
  // Recurrent layers

  template<typename T>
  EmbeddingLayer<T>::EmbeddingLayer(int var_row_index, MatrixX<T> W):
    m_var_row_index(var_row_index),
    m_W(W)
  {
    if(var_row_index < 0)
      throw NNConfigurationException(
        "EmbeddingLayer::EmbeddingLayer - can not set var_row_index<0,"
        " it is an index for a matrix row!");
  }

  template<typename T>
  MatrixX<T> EmbeddingLayer<T>::scan( const MatrixX<T>& x) const {

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
  LSTMLayer<T>::LSTMLayer(ActivationConfig activation,
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

    m_activation_fun.reset(get_raw_activation_layer<T>(activation));
    m_inner_activation_fun.reset(
      get_raw_activation_layer<T>(inner_activation));
  }

  // internal structure created on each scan call
  template<typename T>
  struct LSTMState {
    LSTMState(std::size_t n_input, std::size_t n_outputs);
    MatrixX<T> C_t;
    MatrixX<T> h_t;
    int time;
  };

  template<typename T>
  LSTMState<T>::LSTMState(std::size_t n_input, std::size_t n_output):
    C_t(MatrixX<T>::Zero(n_output, n_input)),
    h_t(MatrixX<T>::Zero(n_output, n_input)),
    time(0)
  {
  }

  template<typename T>
  void LSTMLayer<T>::step(const VectorX<T>& x_t, LSTMState<T>& s) const {
    // https://github.com/fchollet/keras/blob/master/keras/layers/recurrent.py#L740

    const auto& act_fun = m_activation_fun;
    const auto& in_act_fun = m_inner_activation_fun;

    int tm1 = s.time == 0 ? 0 : s.time - 1;
    VectorX<T> h_tm1 = s.h_t.col(tm1);
    VectorX<T> C_tm1 = s.C_t.col(tm1);

    VectorX<T> i  =  in_act_fun->compute(m_W_i*x_t + m_b_i + m_U_i*h_tm1);
    VectorX<T> f  =  in_act_fun->compute(m_W_f*x_t + m_b_f + m_U_f*h_tm1);
    VectorX<T> o  =  in_act_fun->compute(m_W_o*x_t + m_b_o + m_U_o*h_tm1);
    VectorX<T> ct =  act_fun->compute(m_W_c*x_t + m_b_c + m_U_c*h_tm1);

    s.C_t.col(s.time) = f.cwiseProduct(C_tm1) + i.cwiseProduct(ct);
    s.h_t.col(s.time) = o.cwiseProduct(act_fun->compute(s.C_t.col(s.time)));
  }

  template<typename T>
  MatrixX<T> LSTMLayer<T>::scan( const MatrixX<T>& x ) const {

    LSTMState<T> state(x.cols(), m_n_outputs);

    for(state.time = 0; state.time < x.cols(); state.time++) {
      step( x.col( state.time ), state );
    }

    return state.h_t;
  }


  // GRU layer
  template<typename T>
  GRULayer<T>::GRULayer(ActivationConfig activation,
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

    m_activation_fun.reset(get_raw_activation_layer<T>(activation));
    m_inner_activation_fun.reset(
      get_raw_activation_layer<T>(inner_activation));
  }
  // internal structure created on each scan call
  template<typename T>
  struct GRUState {
    GRUState(std::size_t n_input, std::size_t n_outputs);
    MatrixX<T> h_t;
    int time;
  };

  template<typename T>
  GRUState<T>::GRUState(std::size_t n_input, std::size_t n_output):
    h_t(MatrixX<T>::Zero(n_output, n_input)),
    time(0)
  {
  }

  template<typename T>
  void GRULayer<T>::step( const VectorX<T>& x_t, GRUState<T>& s) const {
    // https://github.com/fchollet/keras/blob/master/keras/layers/recurrent.py#L547

    const auto& act_fun = m_activation_fun;
    const auto& in_act_fun = m_inner_activation_fun;

    int tm1 = s.time == 0 ? 0 : s.time - 1;
    VectorX<T> h_tm1 = s.h_t.col(tm1);
    VectorX<T> z  = in_act_fun->compute(m_W_z*x_t + m_b_z + m_U_z*h_tm1);
    VectorX<T> r  = in_act_fun->compute(m_W_r*x_t + m_b_r + m_U_r*h_tm1);
    VectorX<T> rh = r.cwiseProduct(h_tm1);
    VectorX<T> hh = act_fun->compute(m_W_h*x_t + m_b_h + m_U_h*rh);
    VectorX<T> one = VectorX<T>::Ones(z.size());
    s.h_t.col(s.time)  = z.cwiseProduct(h_tm1) + (one - z).cwiseProduct(hh);
  }



  template<typename T>
  MatrixX<T> GRULayer<T>::scan( const MatrixX<T>& x ) const {

    GRUState<T> state(x.cols(), m_n_outputs);

    for(state.time = 0; state.time < x.cols(); state.time++) {
      step( x.col( state.time ), state );
    }

    return state.h_t;
  }


    // SimpleRNN layer
    template<typename T>
    SimpleRNNLayer<T>::SimpleRNNLayer(ActivationConfig activation,
                         MatrixX<T> W_h, MatrixX<T> U_h, VectorX<T> b_h):
      m_W_h(W_h),
      m_U_h(U_h),
      m_b_h(b_h)
    {
      m_n_outputs = m_W_h.rows();

      m_activation_fun.reset(get_raw_activation_layer<T>(activation));
    }

    // internal structure created on each scan call
    template<typename T>
    struct SimpleRNNState {
      SimpleRNNState(std::size_t n_input, std::size_t n_outputs);
      MatrixX<T> h_t;
      int time;
    };

    template<typename T>
    SimpleRNNState<T>::SimpleRNNState(std::size_t n_input, std::size_t n_output):
      h_t(MatrixX<T>::Zero(n_output, n_input)),
      time(0)
    {
    }

    template<typename T>
    void SimpleRNNLayer<T>::step(const VectorX<T>& x_t, SimpleRNNState<T>& s) const {
      // https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/keras/layers/recurrent.py#L1376

      const auto& act_fun = m_activation_fun;

      int tm1 = s.time == 0 ? 0 : s.time - 1;
      VectorX<T> h_tm1 = s.h_t.col(tm1);

      s.h_t.col(s.time) = act_fun->compute(m_W_h*x_t + m_b_h + m_U_h*h_tm1);
    }

    template<typename T>
    MatrixX<T> SimpleRNNLayer<T>::scan( const MatrixX<T>& x ) const {

      SimpleRNNState<T> state(x.cols(), m_n_outputs);

      for(state.time = 0; state.time < x.cols(); state.time++) {
        step( x.col( state.time ), state );
      }

      return state.h_t;
    }

    // Conv1d layer
    template<typename T>
    Conv1dLayer<T>::Conv1dLayer(ActivationConfig activation,
                         const MatrixX<T>& W_h, const VectorX<T>& b_h,
                         lwt::Conv1dConfig cfg):
      m_W_h(W_h),
      m_b_h(b_h),
      m_dilation_rate(cfg.dilation_rate),
      m_padding(cfg.padding)
    {
      m_n_outputs = m_b_h.size();
      m_kernel_size = W_h.rows()/m_n_outputs;
      m_n_inputs = W_h.cols();

      m_activation_fun.reset(get_raw_activation_layer<T>(activation));
    }

    template<typename T>
    MatrixX<T> Conv1dLayer<T>::scan( const MatrixX<T>& x ) const {
      std::size_t seq_length;
      MatrixX<T> x_p;
      if(m_padding == lwt::Padding::CAUSAL){
        seq_length = x.cols();
        x_p = MatrixX<T>::Zero(x.rows(), seq_length + (m_kernel_size - 1)*m_dilation_rate);
        x_p.block(0, (m_kernel_size - 1)*m_dilation_rate, x.rows(), seq_length) = x;
      }
      else if(m_padding == lwt::Padding::SAME){
        seq_length = x.cols();
        x_p = MatrixX<T>::Zero(x.rows(), seq_length + (m_kernel_size - 1)*m_dilation_rate);
        // padding favoring start if kernel is even
        x_p.block(0, ((m_kernel_size - 1)*m_dilation_rate + 1)/2, x.rows(), seq_length) = x;
      }
      else{
        x_p = x; // valid/no padding
        // Sequence gets shortened by field of view - 1
        seq_length = x.cols() - (m_kernel_size - 1)*m_dilation_rate;
        assert(seq_length >= 1);
      }

      MatrixX<T> result = MatrixX<T>::Zero(m_n_outputs, seq_length);
      MatrixX<T> result_act = MatrixX<T>::Zero(m_n_outputs, seq_length);
      for(std::size_t o=0; o < m_n_outputs; o++){
        result.row(o) = MatrixX<T>::Constant(1, seq_length, m_b_h[o]);
      }

      MatrixX<T> k_prod = m_W_h * x_p;
      for(std::size_t o=0; o<m_n_outputs; o++){
        for(std::size_t k=0; k<m_kernel_size; k++){
          result.row(o) += k_prod.block(o*m_kernel_size + k, k*m_dilation_rate, 1, seq_length);
        }
        result_act.row(o) = m_activation_fun->compute(result.row(o));
      }

      return result_act;
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
  ILayer<T>* get_raw_activation_layer(ActivationConfig activation) {
    // Check for special cases. If it's not one, use
    // UnaryActivationLayer
    switch (activation.function) {
    case Activation::LINEAR: return new DummyLayer<T>;
    case Activation::SOFTMAX: return new SoftmaxLayer<T>;
    case Activation::RECTIFIED: return new ReLULayer<T>;
    case Activation::SIGMOID: return new SigmoidLayer<T>;
    case Activation::HARD_SIGMOID: return new HardSigmoidLayer<T>;
    case Activation::TANH: return new TanhLayer<T>;
    case Activation::ABS: return new AbsLayer<T>;
    default: return new UnaryActivationLayer<T>(activation);
    }
  }

  // Most activation functions should be handled here.
  template<typename T>
  std::function<T(T)> get_activation(lwt::ActivationConfig act) {
    using namespace lwt;
    // the first block here is legacy layers
    switch (act.function) {
    case Activation::SIGMOID_LEGACY: return nn_sigmoid<T>;
    case Activation::HARD_SIGMOID_LEGACY: return nn_hard_sigmoid<T>;
    case Activation::TANH_LEGACY: return nn_tanh<T>;
    case Activation::RECTIFIED_LEGACY: return nn_relu<T>;
      // at some point we should probably port these to be ILayer too
    case Activation::SWISH: return Swish<T>(act.alpha);
    case Activation::ELU: return ELU<T>(act.alpha);
    case Activation::LEAKY_RELU: return LeakyReLU<T>(act.alpha);
    case Activation::ABS: return [](T x){using std::abs; /* autodiff... */ return abs(x);};
    default: {
      throw NNConfigurationException("Got undefined activation function");
    }
    }
  }

  // Note: this is deprecated in favor of SigmoidLayer, still here for
  // benchmarking comparisons.
  template<typename T>
  T nn_sigmoid( T x ){
    //github.com/Theano/Theano/blob/master/theano/tensor/nnet/sigm.py#L35
    using std::exp;  // strange requirement from autodiff
    if (x < -30.0) return 0.0;
    if (x >  30.0) return 1.0;
    return 1.0 / (1.0 + exp(-1.0*x));
  }

  // Note: this is deprecated in favor of HardSigmoidLayer, still here for
  // benchmarking comparisons.
  template<typename T>
  T nn_hard_sigmoid( T x ){
    //github.com/Theano/Theano/blob/master/theano/tensor/nnet/sigm.py#L279
    T out = 0.2*x + 0.5;
    if (out < 0) return 0.0;
    if (out > 1) return 1.0;
    return out;
  }

  template<typename T>
  Swish<T>::Swish(T alpha):
    m_alpha(alpha)
  {}

  template<typename T>
  T Swish<T>::operator()(T x) const {
    return x * nn_sigmoid<T>(m_alpha * x);
  }

  // Note: this is deprecated in favor of TanhLayer, still here for
  // benchmarking comparisons.
  template<typename T>
  T nn_tanh( T x ){
    using std::tanh; // strange requirement from autodiff
    return tanh(x);
  }

  // Note: this is deprecated in favor of ReLULayer, still here for
  // benchmarking comparisons.
  template<typename T>
  T nn_relu( T x) {
    using std::isnan;
    if (isnan(x)) return x;
    else return x > 0 ? x : 0;
  }

  template<typename T>
  ELU<T>::ELU(T alpha):
    m_alpha(alpha)
  {}

  template<typename T>
  T ELU<T>::operator()( T x ) const {
    /* ELU function : https://arxiv.org/pdf/1511.07289.pdf
       f(x)=(x>=0)*x + ( (x<0)*alpha*(exp(x)-1) )
    */
    using std::exp; // strange requirement from autodiff
    T exp_term = m_alpha * (exp(x)-1);
    return x>=0 ? x : exp_term;
  }

  template<typename T>
  LeakyReLU<T>::LeakyReLU(T alpha):
    m_alpha(alpha)
  {}

  template<typename T>
  T LeakyReLU<T>::operator()(T x) const {
    return x > 0 ? x : static_cast<T>(m_alpha * x); // weird autodiff
  }

  // ________________________________________________________________________
  // utility functions

  // check to see if the base type of the NN can be assigned to
  template<typename To, typename From>
  struct conversion_check {
    static const bool value =
      std::is_same<To, From>::value ||
      std::is_assignable<To, From>::value ||
      std::is_convertible<From, To>::value;
  };

  template<typename T1, typename T2>
  MatrixX<T1> build_matrix(const std::vector<T2>& weights, std::size_t n_inputs)
  {
    static_assert( conversion_check<T1,T2>::value,
                   "T2 cannot be implicitly assigned to T1" );

    std::size_t n_elements = weights.size();
    if ((n_elements % n_inputs) != 0) {
      std::string problem = "matrix elements not divisible by number"
        " of columns. Elements: " + std::to_string(n_elements) +
        ", Inputs: " + std::to_string(n_inputs);
      throw lwt::NNConfigurationException(problem);
    }
    std::size_t n_outputs = n_elements / n_inputs;
    MatrixX<T1> matrix(n_outputs, n_inputs);
    for (std::size_t row = 0; row < n_outputs; row++) {
      for (std::size_t col = 0; col < n_inputs; col++) {
        T1 element = weights.at(col + row * n_inputs);
        matrix(row, col) = element;
      }
    }
    return matrix;
  }

  template<typename T1, typename T2>
  VectorX<T1> build_vector(const std::vector<T2>& bias)
  {
    static_assert( conversion_check<T1,T2>::value,
                   "T2 cannot be implicitly assigned to T1" );

    VectorX<T1> out(bias.size());
    std::size_t idx = 0;
    for (const auto& val: bias) {
      out(idx) = val;
      idx++;
    }
    return out;
  }

  // component-wise getters (for Highway, lstm, etc)
  template<typename T>
  DenseComponents<T> get_component(const lwt::LayerConfig& layer, std::size_t n_in) {
    using namespace Eigen;
    using namespace lwt;
    MatrixX<T> weights = build_matrix<T, double>(layer.weights, n_in);
    std::size_t n_out = weights.rows();
    VectorX<T> bias = build_vector<T, double>(layer.bias);

    // the u element is optional
    std::size_t u_el = layer.U.size();
    MatrixX<T> U = u_el ? build_matrix<T, double>(layer.U, n_out) : MatrixX<T>::Zero(0,0);

    std::size_t u_out = U.rows();
    std::size_t b_out = bias.rows();
    bool u_mismatch = (u_out != n_out) && (u_out > 0);
    if ( u_mismatch || b_out != n_out) {
      throw NNConfigurationException(
        "Output dims mismatch, W: " + std::to_string(n_out) +
        ", U: " + std::to_string(u_out) + ", b: " + std::to_string(b_out));
    }
    return {weights, U, bias};
  }

} // namespace generic
} // namespace lwt

#endif
