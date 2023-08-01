#ifndef LWTNN_GENERIC_STACK_HH
#define LWTNN_GENERIC_STACK_HH

// Stack classes
//
// These are the low-level classes that implement feed-forward and
// recurrent neural networks. All the Eigen-dependant code in this
// library should live in this file.
//
// To keep the Eigen code out of the high-level interface, the STL ->
// Eigen ``preprocessor'' classes are also defined here.
//
// The ordering of classes is as follows:
//  - Feed-forward Stack class
//  - Feed-forward Layer classes
//  - Recurrent Stack class
//  - Recurrent layers
//  - Activation functions
//  - Various utility functions

#include "lwtnn/Exceptions.hh"
#include "lwtnn/NNLayerConfig.hh"
#include "lwtnn/generic/eigen_typedefs.hh"

#include <vector>
#include <functional>
#include <memory>

namespace lwt
{

namespace generic
{

  template<typename T>
  class ILayer;

  template<typename T>
  class IRecurrentLayer;

  // ______________________________________________________________________
  // Feed forward Stack class

  template<typename T>
  class Stack
  {
  public:
    // constructor for dummy net
    Stack();
    // constructor for real net
    Stack(std::size_t n_inputs, const std::vector<LayerConfig>& layers,
          std::size_t skip_layers = 0);
    ~Stack();

    // make non-copyable for now
    Stack(Stack&) = delete;
    Stack& operator=(Stack&) = delete;

    VectorX<T> compute(VectorX<T>) const;
    std::size_t n_outputs() const;

  private:
    // returns the size of the next layer
    std::size_t add_layers(std::size_t n_inputs, const LayerConfig&);
    std::size_t add_dense_layers(std::size_t n_inputs, const LayerConfig&);
    std::size_t add_normalization_layers(std::size_t n_inputs, const LayerConfig&);
    std::size_t add_highway_layers(std::size_t n_inputs, const LayerConfig&);
    std::size_t add_maxout_layers(std::size_t n_inputs, const LayerConfig&);
    std::vector<ILayer<T>*> m_layers;
    std::size_t m_n_outputs;
  };

  // _______________________________________________________________________
  // Feed-forward layers

  template<typename T>
  class ILayer
  {
  public:
    virtual ~ILayer() {}
    virtual VectorX<T> compute(const VectorX<T>&) const = 0;
  };

  template<typename T>
  class DummyLayer: public ILayer<T>
  {
  public:
    virtual VectorX<T> compute(const VectorX<T>&) const override;
  };

  template<typename T>
  class UnaryActivationLayer: public ILayer<T>
  {
  public:
    UnaryActivationLayer(ActivationConfig);
    virtual VectorX<T> compute(const VectorX<T>&) const override;
  private:
    std::function<T(T)> m_func;
  };

  template<typename T>
  class SoftmaxLayer: public ILayer<T>
  {
  public:
    virtual VectorX<T> compute(const VectorX<T>&) const override;
  };

  template<typename T>
  class ReLULayer: public ILayer<T>
  {
    virtual VectorX<T> compute(const VectorX<T>&) const override;
  };

  template<typename T>
  class SigmoidLayer: public ILayer<T>
  {
    virtual VectorX<T> compute(const VectorX<T>&) const override;
  };

  template<typename T>
  class HardSigmoidLayer: public ILayer<T>
  {
    virtual VectorX<T> compute(const VectorX<T>&) const override;
  };

  template<typename T>
  class TanhLayer: public ILayer<T>
  {
    virtual VectorX<T> compute(const VectorX<T>&) const override;
  };

  template<typename T>
  class AbsLayer: public ILayer<T>
  {
    virtual VectorX<T> compute(const VectorX<T>&) const override;
  };

  template<typename T>
  class BiasLayer: public ILayer<T>
  {
  public:
    BiasLayer(const VectorX<T>& bias);
    template<typename U> BiasLayer(const std::vector<U>& bias);
    virtual VectorX<T> compute(const VectorX<T>&) const override;
  private:
    VectorX<T> m_bias;
  };

  template<typename T>
  class MatrixLayer: public ILayer<T>
  {
  public:
    MatrixLayer(const MatrixX<T>& matrix);
    virtual VectorX<T> compute(const VectorX<T>&) const override;
  private:
    MatrixX<T> m_matrix;
  };

  template<typename T>
  class MaxoutLayer: public ILayer<T>
  {
  public:
    typedef std::pair<MatrixX<T>, VectorX<T>> InitUnit;
    MaxoutLayer(const std::vector<InitUnit>& maxout_tensor);
    virtual VectorX<T> compute(const VectorX<T>&) const override;
  private:
    std::vector<MatrixX<T>> m_matrices;
    MatrixX<T> m_bias;
  };

  /// Normalization layer ///
  /// https://arxiv.org/abs/1502.03167 ///
  template<typename T>
  class NormalizationLayer : public ILayer<T>
  {
  public:
    NormalizationLayer(const VectorX<T>& W,const VectorX<T>& b);
    virtual VectorX<T> compute(const VectorX<T>&) const override;

  private:
    VectorX<T> _W;
    VectorX<T> _b;

  };

  //http://arxiv.org/pdf/1505.00387v2.pdf
  template<typename T>
  class HighwayLayer: public ILayer<T>
  {
  public:
    HighwayLayer(const MatrixX<T>& W,
                 const VectorX<T>& b,
                 const MatrixX<T>& W_carry,
                 const VectorX<T>& b_carry,
                 ActivationConfig activation);
    virtual VectorX<T> compute(const VectorX<T>&) const override;
  private:
    MatrixX<T> m_w_t;
    VectorX<T> m_b_t;
    MatrixX<T> m_w_c;
    VectorX<T> m_b_c;
    std::unique_ptr<ILayer<T>> m_act;
  };

  // ______________________________________________________________________
  // Recurrent StackT

  template<typename T>
  class RecurrentStack
  {
  public:
    RecurrentStack(std::size_t n_inputs, const std::vector<LayerConfig>& layers);
    ~RecurrentStack();
    RecurrentStack(RecurrentStack&) = delete;
    RecurrentStack& operator=(RecurrentStack&) = delete;
    MatrixX<T> scan(MatrixX<T> inputs) const;
    std::size_t n_outputs() const;
  private:
    std::vector<IRecurrentLayer<T>*> m_layers;
    std::size_t add_lstm_layers(std::size_t n_inputs, const LayerConfig&);
    std::size_t add_gru_layers(std::size_t n_inputs, const LayerConfig&);
    std::size_t add_simplernn_layers(std::size_t n_inputs, const LayerConfig&);
    std::size_t add_conv1d_layers(std::size_t n_inputs, const LayerConfig&);
    std::size_t add_embedding_layers(std::size_t n_inputs, const LayerConfig&);
    std::size_t m_n_outputs;
  };

  // This is the old RecurrentStack. Should probably absorb this into
  // the high-level interface in LightweightRNN, since all it does is
  // provide a slightly higher-level interface to a network which
  // combines recurrent + ff layers.
  template<typename T>
  class ReductionStack
  {
  public:
    ReductionStack(std::size_t n_in, const std::vector<LayerConfig>& layers);
    ~ReductionStack();
    ReductionStack(ReductionStack&) = delete;
    ReductionStack& operator=(ReductionStack&) = delete;
    VectorX<T> reduce(MatrixX<T> inputs) const;
    std::size_t n_outputs() const;
  private:
    RecurrentStack<T>* m_recurrent;
    Stack<T>* m_stack;
  };

  // __________________________________________________________________
  // Recurrent layers

  template<typename T>
  class IRecurrentLayer
  {
  public:
    virtual ~IRecurrentLayer() {}
    virtual MatrixX<T> scan( const MatrixX<T>&) const = 0;
  };

  template<typename T>
  class EmbeddingLayer : public IRecurrentLayer<T>
  {
  public:
    EmbeddingLayer(int var_row_index, MatrixX<T> W);
    virtual ~EmbeddingLayer() {};
    virtual MatrixX<T> scan( const MatrixX<T>&) const override;

  private:
    int m_var_row_index;
    MatrixX<T> m_W;
  };

  /// long short term memory ///
  template<typename T> struct LSTMState;

  template<typename T>
  class LSTMLayer : public IRecurrentLayer<T>
  {
  public:
    LSTMLayer(ActivationConfig activation,
              ActivationConfig inner_activation,
              MatrixX<T> W_i, MatrixX<T> U_i, VectorX<T> b_i,
              MatrixX<T> W_f, MatrixX<T> U_f, VectorX<T> b_f,
              MatrixX<T> W_o, MatrixX<T> U_o, VectorX<T> b_o,
              MatrixX<T> W_c, MatrixX<T> U_c, VectorX<T> b_c);

    virtual ~LSTMLayer() {};
    virtual MatrixX<T> scan( const MatrixX<T>&) const override;
    void step( const VectorX<T>& input, LSTMState<T>& ) const;

  private:
    std::unique_ptr<ILayer<T>> m_activation_fun;
    std::unique_ptr<ILayer<T>> m_inner_activation_fun;

    MatrixX<T> m_W_i;
    MatrixX<T> m_U_i;
    VectorX<T> m_b_i;

    MatrixX<T> m_W_f;
    MatrixX<T> m_U_f;
    VectorX<T> m_b_f;

    MatrixX<T> m_W_o;
    MatrixX<T> m_U_o;
    VectorX<T> m_b_o;

    MatrixX<T> m_W_c;
    MatrixX<T> m_U_c;
    VectorX<T> m_b_c;

    int m_n_outputs;
  };


  /// gated recurrent unit ///
  template<typename T> struct GRUState;

  template<typename T>
  class GRULayer : public IRecurrentLayer<T>
  {
  public:
    GRULayer(ActivationConfig activation,
             ActivationConfig inner_activation,
             MatrixX<T> W_z, MatrixX<T> U_z, VectorX<T> b_z,
             MatrixX<T> W_r, MatrixX<T> U_r, VectorX<T> b_r,
             MatrixX<T> W_h, MatrixX<T> U_h, VectorX<T> b_h);

    virtual ~GRULayer() {};
    virtual MatrixX<T> scan( const MatrixX<T>&) const override;
    void step( const VectorX<T>& input, GRUState<T>& ) const;

  private:
    std::unique_ptr<ILayer<T>> m_activation_fun;
    std::unique_ptr<ILayer<T>> m_inner_activation_fun;

    MatrixX<T> m_W_z;
    MatrixX<T> m_U_z;
    VectorX<T> m_b_z;

    MatrixX<T> m_W_r;
    MatrixX<T> m_U_r;
    VectorX<T> m_b_r;

    MatrixX<T> m_W_h;
    MatrixX<T> m_U_h;
    VectorX<T> m_b_h;

    int m_n_outputs;
  };

  template<typename T> struct SimpleRNNState;

  template<typename T>
  class SimpleRNNLayer : public IRecurrentLayer<T>
  {
  public:
    SimpleRNNLayer(ActivationConfig activation,
              MatrixX<T> W_h, MatrixX<T> U_h, VectorX<T> b_h);

    virtual ~SimpleRNNLayer() {};
    virtual MatrixX<T> scan( const MatrixX<T>&) const override;
    void step( const VectorX<T>& input, SimpleRNNState<T>& ) const;

  private:
    std::unique_ptr<ILayer<T>> m_activation_fun;

    MatrixX<T> m_W_h;
    MatrixX<T> m_U_h;
    VectorX<T> m_b_h;

    int m_n_outputs;
  };

  template<typename T>
  class Conv1dLayer : public IRecurrentLayer<T>
  {
  public:
    Conv1dLayer(ActivationConfig activation,
              const MatrixX<T>& W_h, const VectorX<T>& b_h, lwt::Conv1dConfig cfg);

    virtual ~Conv1dLayer() {};
    virtual MatrixX<T> scan( const MatrixX<T>&) const override;

  private:
    std::unique_ptr<ILayer<T>> m_activation_fun;

    MatrixX<T> m_W_h;
    VectorX<T> m_b_h;

    std::size_t m_n_outputs;
    std::size_t m_n_inputs;
    std::size_t m_kernel_size;
    std::size_t m_dilation_rate;
    lwt::Padding m_padding;

  };


  // ______________________________________________________________________
  // Activation functions

  // note that others are supported but are too simple to
  // require a special function
  template<typename T> T nn_sigmoid( T x );
  template<typename T> T nn_hard_sigmoid( T x );
  template<typename T> T nn_tanh( T x );
  template<typename T> T nn_relu( T x );

  template<typename T>
  class ELU
  {
  public:
    ELU(T alpha);
    T operator()(T) const;
  private:
    T m_alpha;
  };


  template<typename T>
  class LeakyReLU
  {
  public:
    LeakyReLU(T alpha);
    T operator()(T) const;
  private:
    T m_alpha;
  };

  template<typename T>
  class Swish
  {
  public:
    Swish(T alpha);
    T operator()(T) const;
  private:
    T m_alpha;
  };

  template<typename T> std::function<T(T)> get_activation(lwt::ActivationConfig);

  // WARNING: you own this pointer! Only call when assigning to member data!
  template<typename T> ILayer<T>* get_raw_activation_layer(ActivationConfig);

  // ______________________________________________________________________
  // utility functions

  // functions to build up basic units from vectors
  template<typename T1, typename T2> MatrixX<T1> build_matrix(const std::vector<T2>& weights, std::size_t n_inputs);
  template<typename T1, typename T2> VectorX<T1> build_vector(const std::vector<T2>& bias);

  // LSTM component for convenience in some layers
  template<typename T>
  struct DenseComponents
  {
    MatrixX<T> W;
    MatrixX<T> U;
    VectorX<T> b;
  };

  template<typename T> DenseComponents<T> get_component(const lwt::LayerConfig& layer, std::size_t n_in);

} // namespace generic

  // consistency checks
  void throw_if_not_maxout(const LayerConfig& layer);
  void throw_if_not_dense(const LayerConfig& layer);
  void throw_if_not_conv1d(const LayerConfig& layer, std::size_t n_inputs);
  void throw_if_not_normalization(const LayerConfig& layer);

} // namespace lwt

#include "Stack.tcc"

#endif // LWTNN_GENERIC_STACK_HH
