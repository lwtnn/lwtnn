#ifndef STACK_HH
#define STACK_HH

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
//  - RecurrentStack class
//  - Recurrent layers
//  - Activation functions
//  - Various utility functions


#include "Exceptions.hh"
#include "NNLayerConfig.hh"

#include <Eigen/Dense>

#include <vector>
#include <functional>

namespace lwt {

  using Eigen::VectorXd;
  using Eigen::MatrixXd;

  class ILayer;
  class IRecurrentLayer;


  // ______________________________________________________________________
  // Feed forward Stack class

  class Stack
  {
  public:
    // constructor for dummy net
    Stack();
    // constructor for real net
    Stack(size_t n_inputs, const std::vector<LayerConfig>& layers,
          size_t skip_layers = 0);
    ~Stack();

    // make non-copyable for now
    Stack(Stack&) = delete;
    Stack& operator=(Stack&) = delete;

    VectorXd compute(VectorXd) const;
    size_t n_outputs() const;

  private:
    // returns the size of the next layer
    size_t add_layers(size_t n_inputs, const LayerConfig&);
    size_t add_dense_layers(size_t n_inputs, const LayerConfig&);
    size_t add_normalization_layers(size_t n_inputs, const LayerConfig&);
    size_t add_highway_layers(size_t n_inputs, const LayerConfig&);
    size_t add_maxout_layers(size_t n_inputs, const LayerConfig&);
    std::vector<ILayer*> m_layers;
    size_t m_n_outputs;
  };

  // _______________________________________________________________________
  // Feed-forward layers

  class ILayer
  {
  public:
    virtual ~ILayer() {}
    virtual VectorXd compute(const VectorXd&) const = 0;
  };

  class DummyLayer: public ILayer
  {
  public:
    virtual VectorXd compute(const VectorXd&) const override;
  };

  class UnaryActivationLayer: public ILayer
  {
  public:
    UnaryActivationLayer(ActivationConfig);
    virtual VectorXd compute(const VectorXd&) const override;
  private:
    std::function<double(double)> m_func;
  };

  class SoftmaxLayer: public ILayer
  {
  public:
    virtual VectorXd compute(const VectorXd&) const override;
  };

  class BiasLayer: public ILayer
  {
  public:
    BiasLayer(const VectorXd& bias);
    BiasLayer(const std::vector<double>& bias);
    virtual VectorXd compute(const VectorXd&) const override;
  private:
    VectorXd m_bias;
  };

  class MatrixLayer: public ILayer
  {
  public:
    MatrixLayer(const MatrixXd& matrix);
    virtual VectorXd compute(const VectorXd&) const override;
  private:
    MatrixXd m_matrix;
  };

  class MaxoutLayer: public ILayer
  {
  public:
    typedef std::pair<MatrixXd, VectorXd> InitUnit;
    MaxoutLayer(const std::vector<InitUnit>& maxout_tensor);
    virtual VectorXd compute(const VectorXd&) const override;
  private:
    std::vector<MatrixXd> m_matrices;
    MatrixXd m_bias;
  };


  /// Normalization layer ///
  /// https://arxiv.org/abs/1502.03167 ///
  class NormalizationLayer : public ILayer
  {

  public:
    NormalizationLayer(const VectorXd& W,const VectorXd& b);
    virtual VectorXd compute(const VectorXd&) const override;

  private:
    VectorXd _W;
    VectorXd _b;

  };

  //http://arxiv.org/pdf/1505.00387v2.pdf
  class HighwayLayer: public ILayer
  {
  public:
    HighwayLayer(const MatrixXd& W,
                 const VectorXd& b,
                 const MatrixXd& W_carry,
                 const VectorXd& b_carry,
                 ActivationConfig activation);
    virtual VectorXd compute(const VectorXd&) const override;
  private:
    MatrixXd m_w_t;
    VectorXd m_b_t;
    MatrixXd m_w_c;
    VectorXd m_b_c;
    std::function<double(double)> m_act;
  };

  // ______________________________________________________________________
  // Recurrent Stack

  class RecurrentStack
  {
  public:
    RecurrentStack(size_t n_inputs, const std::vector<LayerConfig>& layers);
    ~RecurrentStack();
    RecurrentStack(RecurrentStack&) = delete;
    RecurrentStack& operator=(RecurrentStack&) = delete;
    MatrixXd scan(MatrixXd inputs) const;
    size_t n_outputs() const;
  private:
    std::vector<IRecurrentLayer*> m_layers;
    size_t add_lstm_layers(size_t n_inputs, const LayerConfig&);
    size_t add_gru_layers(size_t n_inputs, const LayerConfig&);
    size_t add_embedding_layers(size_t n_inputs, const LayerConfig&);
    size_t m_n_outputs;
  };

  // This is the old RecurrentStack. Should probably absorb this into
  // the high-level interface in LightweightRNN, since all it does is
  // provide a slightly higher-level interface to a network which
  // combines recurrent + ff layers.
  class ReductionStack
  {
  public:
    ReductionStack(size_t n_in, const std::vector<LayerConfig>& layers);
    ~ReductionStack();
    ReductionStack(ReductionStack&) = delete;
    ReductionStack& operator=(ReductionStack&) = delete;
    VectorXd reduce(MatrixXd inputs) const;
    size_t n_outputs() const;
  private:
    RecurrentStack* m_recurrent;
    Stack* m_stack;
  };

  // __________________________________________________________________
  // Recurrent layers

  class IRecurrentLayer
  {
  public:
    virtual ~IRecurrentLayer() {}
    virtual MatrixXd scan( const MatrixXd&) const = 0;
  };

  class EmbeddingLayer : public IRecurrentLayer
  {
  public:
    EmbeddingLayer(int var_row_index, MatrixXd W);
    virtual ~EmbeddingLayer() {};
    virtual MatrixXd scan( const MatrixXd&) const override;

  private:
    int m_var_row_index;
    MatrixXd m_W;
  };

  /// long short term memory ///
  struct LSTMState;
  class LSTMLayer : public IRecurrentLayer
  {
  public:
    LSTMLayer(ActivationConfig activation,
              ActivationConfig inner_activation,
              MatrixXd W_i, MatrixXd U_i, VectorXd b_i,
              MatrixXd W_f, MatrixXd U_f, VectorXd b_f,
              MatrixXd W_o, MatrixXd U_o, VectorXd b_o,
              MatrixXd W_c, MatrixXd U_c, VectorXd b_c);

    virtual ~LSTMLayer() {};
    virtual MatrixXd scan( const MatrixXd&) const override;
    void step( const VectorXd& input, LSTMState& ) const;

  private:
    std::function<double(double)> m_activation_fun;
    std::function<double(double)> m_inner_activation_fun;

    MatrixXd m_W_i;
    MatrixXd m_U_i;
    VectorXd m_b_i;

    MatrixXd m_W_f;
    MatrixXd m_U_f;
    VectorXd m_b_f;

    MatrixXd m_W_o;
    MatrixXd m_U_o;
    VectorXd m_b_o;

    MatrixXd m_W_c;
    MatrixXd m_U_c;
    VectorXd m_b_c;

    int m_n_outputs;
  };

  /// gated recurrent unit ///
  struct GRUState;
  class GRULayer : public IRecurrentLayer
  {
  public:
    GRULayer(ActivationConfig activation,
             ActivationConfig inner_activation,
             MatrixXd W_z, MatrixXd U_z, VectorXd b_z,
             MatrixXd W_r, MatrixXd U_r, VectorXd b_r,
             MatrixXd W_h, MatrixXd U_h, VectorXd b_h);

    virtual ~GRULayer() {};
    virtual MatrixXd scan( const MatrixXd&) const override;
    void step( const VectorXd& input, GRUState& ) const;

  private:
    std::function<double(double)> m_activation_fun;
    std::function<double(double)> m_inner_activation_fun;

    MatrixXd m_W_z;
    MatrixXd m_U_z;
    VectorXd m_b_z;

    MatrixXd m_W_r;
    MatrixXd m_U_r;
    VectorXd m_b_r;

    MatrixXd m_W_h;
    MatrixXd m_U_h;
    VectorXd m_b_h;

    int m_n_outputs;
  };

  // ______________________________________________________________________
  // Activation functions

  // note that others are supported but are too simple to
  // require a special function
  double nn_sigmoid( double x );
  double nn_hard_sigmoid( double x );
  double nn_tanh( double x );
  double nn_relu( double x );
  class ELU
  {
  public:
    ELU(double alpha);
    double operator()(double) const;
  private:
    double m_alpha;
  };
  class LeakyReLU
  {
  public:
    LeakyReLU(double alpha);
    double operator()(double) const;
  private:
    double m_alpha;
  };
  class Swish
  {
  public:
    Swish(double alpha);
    double operator()(double) const;
  private:
    double m_alpha;
  };
  std::function<double(double)> get_activation(lwt::ActivationConfig);

  // WARNING: you own this pointer! Only call when assigning to member data!
  ILayer* get_raw_activation_layer(ActivationConfig);

  // ______________________________________________________________________
  // utility functions

  // functions to build up basic units from vectors
  MatrixXd build_matrix(const std::vector<double>& weights, size_t n_inputs);
  VectorXd build_vector(const std::vector<double>& bias);

  // consistency checks
  void throw_if_not_maxout(const LayerConfig& layer);
  void throw_if_not_dense(const LayerConfig& layer);
  void throw_if_not_normalization(const LayerConfig& layer);

  // LSTM component for convenience in some layers
  struct DenseComponents
  {
    Eigen::MatrixXd W;
    Eigen::MatrixXd U;
    Eigen::VectorXd b;
  };
  DenseComponents get_component(const lwt::LayerConfig& layer, size_t n_in);


}

#endif // STACK_HH
