#ifndef NN_LAYER_CONFIG_HH
#define NN_LAYER_CONFIG_HH

// Layer Configiruation for Lightweight Tagger
//
// The structures below are used to initalize
// `LightweightNeuralNetwork` and the simpler `Stack`.
//
// Author: Dan Guest <dguest@cern.ch>

#include <vector>
#include <string>
#include <map>

namespace lwt {
  enum class Activation {LINEAR, SIGMOID, RECTIFIED, SOFTMAX, TANH};
  enum class Architecture {DENSE, MAXOUT, LSTM};
  // components (for LSTM, etc)
  enum class Component {I,O,C,F};
  // const std::vector<Component> LSTM_COMPONENTS {
  //   Component::I, Component::O, Component::C, Component::F};

  struct LayerConfig
  {
    // dense layer info
    std::vector<double> weights;
    std::vector<double> bias;
    std::vector<double> U;      // TODO: what is this thing called in LSTMs?
    Activation activation;

    // additional info for sublayers
    std::vector<LayerConfig> sublayers;
    std::map<Component, LayerConfig> components;

    // arch flag
    Architecture architecture;
  };

  struct Input
  {
    std::string name;
    double offset;
    double scale;
  };
}

#endif
