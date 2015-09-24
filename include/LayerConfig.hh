#ifndef LAYER_CONFIG_HH
#define LAYER_CONFIG_HH

#include <vector>
#include <string>
#include <ostream>

namespace lwt {
  enum class Activation {LINEAR, SIGMOID, RECTIFIED};
  struct LayerConfig
  {
    std::vector<double> weights;
    std::vector<double> bias;
    Activation activation;
  };

  struct Input
  {
    std::string name;
    double offset;
    double scale;
  };
}

#endif
