#ifndef STACK_HH
#define STACK_HH

// 2020: Changes by Benjamin Huth
// - templated all classes


// StackT classes
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
//  - RecurrentStackT class
//  - Recurrent layers
//  - Activation functions
//  - Various utility functions


#include "Exceptions.hh"
#include "NNLayerConfig.hh"

#include <Eigen/Dense>

#include <vector>
#include <functional>

namespace lwt {

  template<typename T>
  using VectorX = Eigen::Matrix<T, Eigen::Dynamic, 1>;
  
  template<typename T>
  using MatrixX = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
  
  template<typename T>
  using ArrayX = Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic>;

  template<typename T>
  class ILayerT;
  
  template<typename T>
  class IRecurrentLayerT;

  class FittableLWTNN;
  
  // ______________________________________________________________________
  // Feed forward Stack class

  template<typename T>
  class StackT
  {
    friend class FittableLWTNN;
      
  public:
    // constructor for dummy net
    StackT();
    // constructor for real net
    StackT(size_t n_inputs, const std::vector<LayerConfig>& layers,
          size_t skip_layers = 0);
    ~StackT();

    // make non-copyable for now
    StackT(StackT&) = delete;
    StackT& operator=(StackT&) = delete;

    VectorX<T> compute(VectorX<T>) const;
    size_t n_outputs() const;

  private:
    // returns the size of the next layer
    size_t add_layers(size_t n_inputs, const LayerConfig&);
    size_t add_dense_layers(size_t n_inputs, const LayerConfig&);
    size_t add_normalization_layers(size_t n_inputs, const LayerConfig&);
    size_t add_highway_layers(size_t n_inputs, const LayerConfig&);
    size_t add_maxout_layers(size_t n_inputs, const LayerConfig&);
    std::vector<ILayerT<T>*> m_layers;
    size_t m_n_outputs;
  };
  
  using Stack = StackT<double>;

  // _______________________________________________________________________
  // Feed-forward layers

  template<typename T>
  class ILayerT
  {
  public:
    virtual ~ILayerT() {}
    virtual VectorX<T> compute(const VectorX<T>&) const = 0;
  };
  
  using ILayer = ILayerT<double>;

  template<typename T>
  class DummyLayerT: public ILayerT<T>
  {
  public:
    virtual VectorX<T> compute(const VectorX<T>&) const override;
  };
  
  using DummyLayer = DummyLayerT<double>;

  template<typename T>
  class UnaryActivationLayerT: public ILayerT<T>
  {
  public:
    UnaryActivationLayerT(ActivationConfig);
    virtual VectorX<T> compute(const VectorX<T>&) const override;
  private:
    std::function<T(T)> m_func;
  };
  
  using UnaryActivationLayer = UnaryActivationLayerT<double>;

  template<typename T>
  class SoftmaxLayerT: public ILayerT<T>
  {
  public:
    virtual VectorX<T> compute(const VectorX<T>&) const override;
  };
  
  using SoftmaxLayer = SoftmaxLayerT<double>;

  template<typename T>
  class BiasLayerT: public ILayerT<T>
  {
    friend class FittableLWTNN;
    
  public:
    BiasLayerT(const VectorX<T>& bias);
    template<typename U> BiasLayerT(const std::vector<U>& bias);
    virtual VectorX<T> compute(const VectorX<T>&) const override;
  private:
    VectorX<T> m_bias;
  };
  
  using BiasLayer = BiasLayerT<double>;

  template<typename T>
  class MatrixLayerT: public ILayerT<T>
  {
    friend class FittableLWTNN;
    
  public:
    MatrixLayerT(const MatrixX<T>& matrix);
    virtual VectorX<T> compute(const VectorX<T>&) const override;
  private:
    MatrixX<T> m_matrix;
  };
  
  using MatrixLayer = MatrixLayerT<double>;

  template<typename T>
  class MaxoutLayerT: public ILayerT<T>
  {
    friend class FittableLWTNN;
    
  public:
    typedef std::pair<MatrixX<T>, VectorX<T>> InitUnit;
    MaxoutLayerT(const std::vector<InitUnit>& maxout_tensor);
    virtual VectorX<T> compute(const VectorX<T>&) const override;
  private:
    std::vector<MatrixX<T>> m_matrices;
    MatrixX<T> m_bias;
  };
  
  using MaxoutLayer = MaxoutLayerT<double>;


  /// Normalization layer ///
  /// https://arxiv.org/abs/1502.03167 ///
  template<typename T>
  class NormalizationLayerT : public ILayerT<T>
  {
    friend class FittableLWTNN;
    
  public:
    NormalizationLayerT(const VectorX<T>& W,const VectorX<T>& b);
    virtual VectorX<T> compute(const VectorX<T>&) const override;

  private:
    VectorX<T> _W;
    VectorX<T> _b;

  };
  
  using NormalizationLayer = NormalizationLayerT<double>;

  //http://arxiv.org/pdf/1505.00387v2.pdf
  template<typename T>
  class HighwayLayerT: public ILayerT<T>
  {
    friend class FittableLWTNN;
    
  public:
    HighwayLayerT(const MatrixX<T>& W,
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
    std::function<T(T)> m_act;
  };
  
  using HighwayLayer = HighwayLayerT<double>;

  // ______________________________________________________________________
  // Recurrent StackT

  template<typename T>
  class RecurrentStackT
  {
  public:
    RecurrentStackT(size_t n_inputs, const std::vector<LayerConfig>& layers);
    ~RecurrentStackT();
    RecurrentStackT(RecurrentStackT&) = delete;
    RecurrentStackT& operator=(RecurrentStackT&) = delete;
    MatrixX<T> scan(MatrixX<T> inputs) const;
    size_t n_outputs() const;
  private:
    std::vector<IRecurrentLayerT<T>*> m_layers;
    size_t add_lstm_layers(size_t n_inputs, const LayerConfig&);
    size_t add_gru_layers(size_t n_inputs, const LayerConfig&);
    size_t add_embedding_layers(size_t n_inputs, const LayerConfig&);
    size_t m_n_outputs;
  };
  
  using RecurrentStack = RecurrentStackT<double>;

  // This is the old RecurrentStack. Should probably absorb this into
  // the high-level interface in LightweightRNN, since all it does is
  // provide a slightly higher-level interface to a network which
  // combines recurrent + ff layers.
  template<typename T>
  class ReductionStackT
  {
  public:
    ReductionStackT(size_t n_in, const std::vector<LayerConfig>& layers);
    ~ReductionStackT();
    ReductionStackT(ReductionStackT&) = delete;
    ReductionStackT& operator=(ReductionStackT&) = delete;
    VectorX<T> reduce(MatrixX<T> inputs) const;
    size_t n_outputs() const;
  private:
    RecurrentStackT<T>* m_recurrent;
    StackT<T>* m_stack;
  };
  
  using ReductionStack = ReductionStackT<double>;

  // __________________________________________________________________
  // Recurrent layers

  template<typename T>
  class IRecurrentLayerT
  {
  public:
    virtual ~IRecurrentLayerT() {}
    virtual MatrixX<T> scan( const MatrixX<T>&) const = 0;
  };
  
  using IRecurrentLayer = IRecurrentLayerT<double>;
  
  template<typename T>
  class EmbeddingLayerT : public IRecurrentLayerT<T>
  {
  public:
    EmbeddingLayerT(int var_row_index, MatrixX<T> W);
    virtual ~EmbeddingLayerT() {};
    virtual MatrixX<T> scan( const MatrixX<T>&) const override;

  private:
    int m_var_row_index;
    MatrixX<T> m_W;
  };
  
  using EmbeddingLayer = EmbeddingLayerT<double>;

  /// long short term memory ///
  template<typename T> struct LSTMStateT;
  using LSTMState = LSTMStateT<double>;
  
  template<typename T>
  class LSTMLayerT : public IRecurrentLayerT<T>
  {
  public:
    LSTMLayerT(ActivationConfig activation,
              ActivationConfig inner_activation,
              MatrixX<T> W_i, MatrixX<T> U_i, VectorX<T> b_i,
              MatrixX<T> W_f, MatrixX<T> U_f, VectorX<T> b_f,
              MatrixX<T> W_o, MatrixX<T> U_o, VectorX<T> b_o,
              MatrixX<T> W_c, MatrixX<T> U_c, VectorX<T> b_c);

    virtual ~LSTMLayerT() {};
    virtual MatrixX<T> scan( const MatrixX<T>&) const override;
    void step( const VectorX<T>& input, LSTMState& ) const;

  private:
    std::function<T(T)> m_activation_fun;
    std::function<T(T)> m_inner_activation_fun;

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
  
  using LSTMLayer = LSTMLayerT<double>;

  /// gated recurrent unit ///
  template<typename T> struct GRUStateT;
  using GRUState = GRUStateT<double>;
  
  template<typename T>
  class GRULayerT : public IRecurrentLayerT<T>
  {
  public:
    GRULayerT(ActivationConfig activation,
             ActivationConfig inner_activation,
             MatrixX<T> W_z, MatrixX<T> U_z, VectorX<T> b_z,
             MatrixX<T> W_r, MatrixX<T> U_r, VectorX<T> b_r,
             MatrixX<T> W_h, MatrixX<T> U_h, VectorX<T> b_h);

    virtual ~GRULayerT() {};
    virtual MatrixX<T> scan( const MatrixX<T>&) const override;
    void step( const VectorX<T>& input, GRUState& ) const;

  private:
    std::function<T(T)> m_activation_fun;
    std::function<T(T)> m_inner_activation_fun;

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
  
  using GRULayer = GRULayerT<double>;

  // ______________________________________________________________________
  // Activation functions

  // note that others are supported but are too simple to
  // require a special function
  template<typename T> T nn_sigmoidT( T x );
  template<typename T> T nn_hard_sigmoidT( T x );
  template<typename T> T nn_tanhT( T x );
  template<typename T> T nn_reluT( T x );
  
  double nn_sigmoid( double x );
  double nn_hard_sigmoid( double x );
  double nn_tanh( double x );
  double nn_relu( double x );
  
  template<typename T>
  class ELUT
  {
  public:
    ELUT(T alpha);
    T operator()(T) const;
  private:
    T m_alpha;
  };
  
  using ELU = ELUT<double>;
  
  template<typename T>
  class LeakyReLUT
  {
  public:
    LeakyReLUT(T alpha);
    T operator()(T) const;
  private:
    T m_alpha;
  };
  
  using LeakyReLU = LeakyReLUT<double>;
  
  template<typename T>
  class SwishT
  {
  public:
    SwishT(T alpha);
    T operator()(T) const;
  private:
    T m_alpha;
  };
  
  using Swish = SwishT<double>;
  
  template<typename T> std::function<T(T)> get_activationT(lwt::ActivationConfig);
  std::function<double(double)> get_activation(lwt::ActivationConfig);

  // WARNING: you own this pointer! Only call when assigning to member data!
  template<typename T> ILayerT<T>* get_raw_activation_layerT(ActivationConfig);
  ILayer* get_raw_activation_layer(ActivationConfig);
  

  // ______________________________________________________________________
  // utility functions

  // functions to build up basic units from vectors
  template<typename T1, typename T2> MatrixX<T1> build_matrixT(const std::vector<T2>& weights, size_t n_inputs);
  template<typename T1, typename T2> VectorX<T1> build_vectorT(const std::vector<T2>& bias);
  
  MatrixX<double> build_matrix(const std::vector<double>& weights, size_t n_inputs);
  VectorX<double> build_vector(const std::vector<double>& bias);

  // consistency checks
  void throw_if_not_maxout(const LayerConfig& layer);
  void throw_if_not_dense(const LayerConfig& layer);
  void throw_if_not_normalization(const LayerConfig& layer);

  // LSTM component for convenience in some layers
  template<typename T>
  struct DenseComponentsT
  {
    MatrixX<T> W;
    MatrixX<T> U;
    VectorX<T> b;
  };
  
  using DenseComponents = DenseComponentsT<double>;
  
  template<typename T> DenseComponentsT<T> get_componentT(const lwt::LayerConfig& layer, size_t n_in);
  DenseComponents get_component(const lwt::LayerConfig& layer, size_t n_in);
}

#include "Stack.txx"

#endif // STACK_HH
