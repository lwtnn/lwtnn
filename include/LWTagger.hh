#ifndef LW_TAGGER_HH
#define LW_TAGGER_HH

#include "LayerConfig.hh"

#include <Eigen/Dense>

#include <vector>
#include <stdexcept>
#include <map>

namespace lwt {

  using Eigen::VectorXd;
  using Eigen::MatrixXd;


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

  // ______________________________________________________________________
  // the NN class

  class Stack
  {
  public:
    // constructor for dummy net
    Stack();
    // constructor for real net
    Stack(size_t n_inputs, const std::vector<LayerConfig>& layers);
    ~Stack();

    // make non-copyable for now
    Stack(Stack&) = delete;
    Stack& operator=(Stack&) = delete;

    VectorXd compute(VectorXd) const;
    size_t n_outputs() const;

  private:
    // returns the size of the next layer
    size_t add_layers(size_t n_inputs, const LayerConfig&);
    std::vector<ILayer*> _layers;
    size_t _n_outputs;
  };

  // ______________________________________________________________________
  // high-level wrapper

  class LWTagger
  {
  public:
    LWTagger(const std::vector<Input>& inputs,
	     const std::vector<LayerConfig>& layers,
	     const std::vector<std::string>& outputs);
    // disable copying until we need it...
    LWTagger(LWTagger&) = delete;
    LWTagger& operator=(LWTagger&) = delete;

    // use a normal map externally, since these are more common in
    // user code.
    // TODO: is it worth changing to unordered_map?
    typedef std::map<std::string, double> ValueMap;
    ValueMap compute(const ValueMap&) const;

  private:
    // use the Stack class above as the computational core
    Stack _stack;

    // input transformations
    VectorXd _offsets;
    VectorXd _scales;
    std::vector<std::string> _names;

    // output labels
    std::vector<std::string> _outputs;

  };
  // ______________________________________________________________________
  // utility functions
  // WARNING: you own this pointer! Only call when assigning to member data!
  ILayer* get_raw_activation_layer(Activation);


  // ______________________________________________________________________
  // exceptions
  // TODO: common base class?
  class NNConfigurationException: public std::logic_error {
  public:
    NNConfigurationException(std::string problem);
  };
  class NNEvaluationException: public std::logic_error {
  public:
    NNEvaluationException(std::string problem);
  };
}
#endif
