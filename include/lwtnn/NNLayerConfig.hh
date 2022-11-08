#ifndef NN_LAYER_CONFIG_HH
#define NN_LAYER_CONFIG_HH

// Layer Configiruation for Lightweight Tagger
//
// The structures below are used to initalize
// `LightweightNeuralNetwork` and the simpler `Stack`.
//
// Author: Dan Guest <dguest@cern.ch>

#include <vector>
#include <map>

namespace lwt {
  enum class Activation {NONE, LINEAR, SIGMOID, RECTIFIED, SOFTMAX, TANH,
    HARD_SIGMOID, ELU, LEAKY_RELU, SWISH, ABS,
    // these "legacy" layers are just around for benchmarking now,
    // will eventually be removed.
    SIGMOID_LEGACY, HARD_SIGMOID_LEGACY, TANH_LEGACY, RECTIFIED_LEGACY};
  enum class Architecture {NONE, DENSE, NORMALIZATION, MAXOUT, HIGHWAY,
      LSTM, GRU, SIMPLERNN, EMBEDDING};
  // components (for LSTM, etc)
  enum class Component {
    I, O, C, F,                 // LSTM
      Z, R, H,                  // GRU
      T, CARRY};                // Highway

  // structure for embedding layers
  struct EmbeddingConfig
  {
    std::vector<double> weights;
    int index;
    int n_out;
  };

  struct ActivationConfig
  {
    Activation function;
    double alpha;
  };

  // main layer configuration
  struct LayerConfig
  {
    // dense layer info
    std::vector<double> weights;
    std::vector<double> bias;
    std::vector<double> U;      // TODO: what is this thing called in LSTMs?
    ActivationConfig activation;
    ActivationConfig inner_activation; // for LSTMs and GRUs

    // additional info for sublayers
    std::vector<LayerConfig> sublayers;
    std::map<Component, LayerConfig> components;
    std::vector<EmbeddingConfig> embedding;

    // arch flag
    Architecture architecture;
  };


  // graph node configuration
  struct NodeConfig
  {
    enum class Type {
      INPUT, INPUT_SEQUENCE, FEED_FORWARD, CONCATENATE, SEQUENCE,
      TIME_DISTRIBUTED, SUM, ADD };
    Type type;
    std::vector<std::size_t> sources;
    int index;                  // input node size, or layer number
  };
}

#endif
