#ifndef PARSE_JSON_HH
#define PARSE_JSON_HH

#include "NNLayerConfig.hh"

#include <istream>

namespace lwt {
  struct JSONConfig
  {
    std::vector<LayerConfig> layers;
    std::vector<Input> inputs;
    std::vector<std::string> outputs;
  };
  JSONConfig parse_json(std::istream& json);
}


#endif
