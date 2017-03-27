#ifndef PARSE_JSON_HH
#define PARSE_JSON_HH

#include "lightweight_network_config.hh"

namespace lwt {
  // build feed forward variant
  JSONConfig parse_json(std::istream& json);
  // build graph variant
  GraphConfig parse_json_graph(std::istream& json);
}


#endif
