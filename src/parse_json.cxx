#include "parse_json.hh"
#include "Streamers.hh"

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <cassert>
#include <string>

#include <iostream>

namespace {
  lwt::Activation get_activation(const std::string&);
}


namespace lwt {

  JSONConfig parse_json(std::istream& json)
  {
    boost::property_tree::ptree pt;
    boost::property_tree::read_json(json, pt);

    JSONConfig cfg;
    for (const auto& v: pt.get_child("inputs")) {
      std::string name = v.second.get<std::string>("name");
      auto offset = v.second.get<double>("offset");
      auto scale = v.second.get<double>("scale");
      Input input{name, offset, scale};
      cfg.inputs.push_back(input);
    }
    for (const auto& v: pt.get_child("layers")) {
      LayerConfig layer;
      for (const auto& wt: v.second.get_child("weights")) {
	layer.weights.push_back(wt.second.get_value<double>());
      }
      for (const auto& bs: v.second.get_child("bias")) {
	layer.bias.push_back(bs.second.get_value<double>());
      }
      layer.activation = get_activation(
	v.second.get<std::string>("activation"));
      cfg.layers.push_back(layer);
    }
    for (const auto& v: pt.get_child("outputs"))
    {
      assert(v.first.empty()); // array elements have no names
      cfg.outputs.push_back(v.second.data());
    }
    return cfg;
  }

}

namespace {
  lwt::Activation get_activation(const std::string& str) {
    using namespace lwt;
    if (str == "linear") return Activation::LINEAR;
    if (str == "sigmoid") return Activation::SIGMOID;
    if (str == "rectified") return Activation::RECTIFIED;
    throw std::logic_error("activation function " + str + " not recognized");
  }
}
