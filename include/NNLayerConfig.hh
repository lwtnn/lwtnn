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

namespace lwt {
  enum class Activation {LINEAR, SIGMOID, RECTIFIED, SOFTMAX};
  enum class Architecture {DENSE, MAXOUT};
  struct LayerConfig
  {
    // dense layer info
    std::vector<double> weights;
    std::vector<double> bias;
    Activation activation;

    // additional info for sublayers
    std::vector<LayerConfig> sublayers;

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
