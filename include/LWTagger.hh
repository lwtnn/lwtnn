#ifndef LW_TAGGER_HH
#define LW_TAGGER_HH

#include "LayerConfig.hh"

#include <Eigen/Dense>

#include <vector>
#include <stdexcept>

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

  private:
    // returns the size of the next layer
    size_t add_layers(size_t n_inputs, const LayerConfig&);
    std::vector<ILayer*> _layers;
  };

  // ______________________________________________________________________
  // utility functions
  // WARNING: you own this pointer! Only call when assigning to member data!
  ILayer* get_raw_activation_layer(Activation);


  // ______________________________________________________________________
  // exceptions
  class NNConfigurationException: public std::logic_error {
  public:
    NNConfigurationException(std::string problem);
  };
}
#endif
