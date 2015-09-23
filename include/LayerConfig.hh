#ifndef LAYER_CONFIG_HH
#define LAYER_CONFIG_HH

#include <vector>

namespace lwt {
  enum class Activation {LINEAR, SIGMOID, RECTIFIED};
  struct LayerConfig
  {
    std::vector<double> weights;
    std::vector<double> bias;
    Activation activation;
  };
}

#endif
