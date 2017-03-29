#ifndef LIGHTWEIGHT_NETWORK_CONFIG_HH
#define LIGHTWEIGHT_NETWORK_CONFIG_HH

// NNLayerConfig is the "low level" configuration, which should
// contain everything needed to create a "Stack" or "Graph".
//
#include "NNLayerConfig.hh"

// The code below is to configure the "high level" interface.

#include <map>
#include <string>

namespace lwt {

  struct Input
  {
    std::string name;
    double offset;
    double scale;
  };


  // feed forward structure
  //
  // Note that this isn't technically JSON-dependant, the name is mostly
  // historical
  struct JSONConfig
  {
    std::vector<LayerConfig> layers;
    std::vector<Input> inputs;
    std::vector<std::string> outputs;
    std::map<std::string, double> defaults;
    std::map<std::string, std::string> miscellaneous;
  };

  // graph structures
  struct InputNodeConfig
  {
    std::string name;
    std::vector<Input> variables;
    std::map<std::string, std::string> miscellaneous;
    std::map<std::string, double> defaults;
  };
  struct OutputNodeConfig
  {
    std::vector<std::string> labels;
    size_t node_index;
  };
  struct GraphConfig
  {
    std::vector<InputNodeConfig> inputs;
    std::vector<InputNodeConfig> input_sequences;
    std::vector<NodeConfig> nodes;
    std::map<std::string, OutputNodeConfig> outputs;
    std::vector<LayerConfig> layers;
  };

}

#endif
