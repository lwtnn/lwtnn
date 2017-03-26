#ifndef PARSE_JSON_HH
#define PARSE_JSON_HH

#include "NNLayerConfig.hh"

#include <istream>
#include <map>

namespace lwt {
  struct JSONConfig
  {
    std::vector<LayerConfig> layers;
    std::vector<Input> inputs;
    std::vector<std::string> outputs;
    std::map<std::string, double> defaults;
    std::map<std::string, std::string> miscellaneous;
  };
  JSONConfig parse_json(std::istream& json);

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
    std::vector<NodeConfig> nodes;
    std::map<std::string, OutputNodeConfig> outputs;
    std::vector<LayerConfig> layers;
  };
  GraphConfig parse_json_graph(std::istream& json);
}


#endif
