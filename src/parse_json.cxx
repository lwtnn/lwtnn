#include "lwtnn/parse_json.hh"

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <cassert>
#include <string>

namespace {
  using namespace boost::property_tree;
  using namespace lwt;
  lwt::Activation get_activation(const std::string&);
  lwt::Architecture get_architecture(const std::string&);
  void add_dense_info(LayerConfig& lc, const ptree::value_type& pt);
  void add_maxout_info(LayerConfig& lc, const ptree::value_type& pt);
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
      layer.activation = get_activation(
        v.second.get<std::string>("activation"));
      layer.architecture = get_architecture(
        v.second.get<std::string>("architecture"));

      if (layer.architecture == Architecture::DENSE) {
        add_dense_info(layer, v);
      } else if (layer.architecture == Architecture::MAXOUT) {
        add_maxout_info(layer, v);
      }
      cfg.layers.push_back(layer);
    }
    for (const auto& v: pt.get_child("outputs"))
    {
      assert(v.first.empty()); // array elements have no names
      cfg.outputs.push_back(v.second.data());
    }
    const std::string dname = "defaults";
    if (pt.count(dname)) {
      for (const auto& def: pt.get_child(dname)) {
        cfg.defaults.emplace(def.first, def.second.get_value<double>());
      }
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
    if (str == "softmax") return Activation::SOFTMAX;
    if (str == "tanh") return Activation::TANH;
    throw std::logic_error("activation function " + str + " not recognized");
    return Activation::LINEAR;
  }

  lwt::Architecture get_architecture(const std::string& str) {
    using namespace lwt;
    if (str == "dense") return Architecture::DENSE;
    if (str == "maxout") return Architecture::MAXOUT;
    throw std::logic_error("architecture " + str + " not recognized");
  }

  void add_dense_info(LayerConfig& layer, const ptree::value_type& v) {
    for (const auto& wt: v.second.get_child("weights")) {
      layer.weights.push_back(wt.second.get_value<double>());
    }
    for (const auto& bs: v.second.get_child("bias")) {
      layer.bias.push_back(bs.second.get_value<double>());
    }
  }

  void add_maxout_info(LayerConfig& layer, const ptree::value_type& v) {
    using namespace lwt;
    for (const auto& sub: v.second.get_child("sublayers")) {
      LayerConfig sublayer;
      add_dense_info(sublayer, sub);
      sublayer.architecture = Architecture::DENSE;
      sublayer.activation = Activation::LINEAR;
      layer.sublayers.push_back(sublayer);
    }
  }

}
