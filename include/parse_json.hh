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
  };
  JSONConfig parse_json(std::istream& json);

  std::map<std::string,double> get_defaults_from_json(std::istream& json);
}


#endif
