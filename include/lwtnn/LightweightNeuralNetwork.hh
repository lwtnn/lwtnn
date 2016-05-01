#ifndef LIGHTWEIGHT_NEURAL_NETWORK_HH
#define LIGHTWEIGHT_NEURAL_NETWORK_HH

// Lightweight Tagger
//
// This is a simple NN implementation, designed to be lightweight in
// terms of both size and dependencies. For sanity we use Eigen, but
// otherwise this aims to be a minimal NN class which is fully
// configurable at runtime.
//
// Author: Dan Guest <dguest@cern.ch>

#include "NNLayerConfig.hh"

#include <Eigen/Dense>

#include <vector>
#include <stdexcept>
#include <map>

namespace lwt {

  using Eigen::VectorXd;
  using Eigen::MatrixXd;

  // use a normal map externally, since these are more common in user
  // code.  TODO: is it worth changing to unordered_map?
  typedef std::map<std::string, double> ValueMap;

  // _______________________________________________________________________
  // layer classes

  class ILayer
  {
  public:
    virtual ~ILayer() {}
    virtual VectorXd compute(const VectorXd&) const = 0;
  };

  class DummyLayer: public ILayer
  {
  public:
    virtual VectorXd compute(const VectorXd&) const;
  };

  class SigmoidLayer: public ILayer
  {
  public:
    virtual VectorXd compute(const VectorXd&) const;
  };
  class RectifiedLayer: public ILayer
  {
  public:
    virtual VectorXd compute(const VectorXd&) const;
  };
  class SoftmaxLayer: public ILayer
  {
  public:
    virtual VectorXd compute(const VectorXd&) const;
  };
  class TanhLayer: public ILayer
  {
  public:
    virtual VectorXd compute(const VectorXd&) const;
  };

  class BiasLayer: public ILayer
  {
  public:
    BiasLayer(const VectorXd& bias);
    BiasLayer(const std::vector<double>& bias);
    virtual VectorXd compute(const VectorXd&) const;
  private:
    VectorXd _bias;
  };

  class MatrixLayer: public ILayer
  {
  public:
    MatrixLayer(const MatrixXd& matrix);
    virtual VectorXd compute(const VectorXd&) const;
  private:
    MatrixXd _matrix;
  };

  class MaxoutLayer: public ILayer
  {
  public:
    typedef std::pair<MatrixXd, VectorXd> InitUnit;
    MaxoutLayer(const std::vector<InitUnit>& maxout_tensor);
    virtual VectorXd compute(const VectorXd&) const;
  private:
    std::vector<MatrixXd> _matrices;
    MatrixXd _bias;
  };

  // ______________________________________________________________________
  // the NN class

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
    size_t add_maxout_layers(size_t n_inputs, const LayerConfig&);
    std::vector<ILayer*> _layers;
    size_t _n_outputs;
  };

  // ______________________________________________________________________
  // input preprocessor (handles normalization and packing into Eigen)

  class InputPreprocessor
  {
  public:
    InputPreprocessor(const std::vector<Input>& inputs);
    VectorXd operator()(const ValueMap&) const;
  private:
    // input transformations
    VectorXd _offsets;
    VectorXd _scales;
    std::vector<std::string> _names;
  };

  // ______________________________________________________________________
  // high-level wrapper

  class LightweightNeuralNetwork
  {
  public:
    LightweightNeuralNetwork(const std::vector<Input>& inputs,
                             const std::vector<LayerConfig>& layers,
                             const std::vector<std::string>& outputs);
    // disable copying until we need it...
    LightweightNeuralNetwork(LightweightNeuralNetwork&) = delete;
    LightweightNeuralNetwork& operator=(LightweightNeuralNetwork&) = delete;

    // use a normal map externally, since these are more common in
    // user code.
    // TODO: is it worth changing to unordered_map?
    ValueMap compute(const ValueMap&) const;

  private:
    // use the Stack class above as the computational core
    Stack _stack;
    InputPreprocessor _preproc;

    // output labels
    std::vector<std::string> _outputs;

  };
  // ______________________________________________________________________
  // utility functions
  // WARNING: you own this pointer! Only call when assigning to member data!
  ILayer* get_raw_activation_layer(Activation);

  // functions to build up basic units from vectors
  MatrixXd build_matrix(const std::vector<double>& weights, size_t n_inputs);
  VectorXd build_vector(const std::vector<double>& bias);

  // consistency checks
  void throw_if_not_maxout(const LayerConfig& layer);
  void throw_if_not_dense(const LayerConfig& layer);

  // ______________________________________________________________________
  // exceptions

  // base exception
  class LightweightNNException: public std::logic_error {
  public:
    LightweightNNException(std::string problem);
  };

  // thrown by the constructor if something goes wrong
  class NNConfigurationException: public LightweightNNException {
  public:
    NNConfigurationException(std::string problem);
  };

  // thrown by `compute`
  class NNEvaluationException: public LightweightNNException {
  public:
    NNEvaluationException(std::string problem);
  };
}
#endif
