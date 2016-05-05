#include "lwtnn/parse_json.hh"

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <cassert>
#include <string>

#include <iostream>

namespace {
  using namespace boost::property_tree;
  using namespace lwt;
  lwt::Activation get_activation(const std::string&);
  lwt::Architecture get_architecture(const std::string&);
  void set_defaults(LayerConfig& lc);
  void add_dense_info(LayerConfig& lc, const ptree::value_type& pt);
  void add_maxout_info(LayerConfig& lc, const ptree::value_type& pt);
  void add_lstm_info(LayerConfig& lc, const ptree::value_type& pt);
  void add_embedding_info(LayerConfig& lc, const ptree::value_type& pt);
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
      set_defaults(layer);
      Architecture arch = get_architecture(
        v.second.get<std::string>("architecture"));

      if (arch == Architecture::DENSE) {
        add_dense_info(layer, v);
      } else if (arch == Architecture::MAXOUT) {
        add_maxout_info(layer, v);
      } else if (arch == Architecture::LSTM) {
        add_lstm_info(layer, v);
      } else if (arch == Architecture::EMBEDDING) {
        add_embedding_info(layer, v);
      } else {
        throw std::logic_error("architecture not implemented");
      }
      layer.architecture = arch;

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
    if (str == "hard_sigmoid") return Activation::HARD_SIGMOID;
    throw std::logic_error("activation function " + str + " not recognized");
    return Activation::LINEAR;
  }


  lwt::Architecture get_architecture(const std::string& str) {
    using namespace lwt;
    if (str == "dense") return Architecture::DENSE;
    if (str == "maxout") return Architecture::MAXOUT;
    if (str == "lstm") return Architecture::LSTM;
    if (str == "embedding") return Architecture::EMBEDDING;
    throw std::logic_error("architecture " + str + " not recognized");
  }

  void set_defaults(LayerConfig& layer) {
    layer.activation = Activation::NONE;
    layer.inner_activation = Activation::NONE;
    layer.architecture = Architecture::NONE;
  }

  void add_dense_info(LayerConfig& layer, const ptree::value_type& v) {
    for (const auto& wt: v.second.get_child("weights")) {
      layer.weights.push_back(wt.second.get_value<double>());
    }
    for (const auto& bs: v.second.get_child("bias")) {
      layer.bias.push_back(bs.second.get_value<double>());
    }
    // this last category is currently only used for LSTM
    if (v.second.count("U") != 0) {
      for (const auto& wt: v.second.get_child("U") ) {
        layer.U.push_back(wt.second.get_value<double>());
      }
    }

    if (v.second.count("activation") != 0) {
      layer.activation = get_activation(
        v.second.get<std::string>("activation"));
    }

  }

  void add_maxout_info(LayerConfig& layer, const ptree::value_type& v) {
    using namespace lwt;
    for (const auto& sub: v.second.get_child("sublayers")) {
      LayerConfig sublayer;
      set_defaults(sublayer);
      add_dense_info(sublayer, sub);
      layer.sublayers.push_back(sublayer);
    }
  }


  const std::map<std::string, lwt::Component> lstm_components {
    {"i", Component::I},
    {"o", Component::O},
    {"c", Component::C},
    {"f", Component::F}
  };

  void add_lstm_info(LayerConfig& layer, const ptree::value_type& v) {
    using namespace lwt;
    for (const auto& comp: v.second.get_child("components")) {
      LayerConfig cfg;
      set_defaults(cfg);
      add_dense_info(cfg, comp);
      layer.components[lstm_components.at(comp.first)] = cfg;
    }
    layer.activation = get_activation(
      v.second.get<std::string>("activation"));
    layer.inner_activation = get_activation(
      v.second.get<std::string>("inner_activation"));
  }


  void add_embedding_info(LayerConfig& layer, const ptree::value_type& v) {
    using namespace lwt;
    for (const auto& sub: v.second.get_child("sublayers")) {
      EmbeddingConfig emb;
      for (const auto& wt: sub.second.get_child("weights")) {
        emb.weights.push_back(wt.second.get_value<double>());
      }
      emb.index = sub.second.get<int>("index");
      emb.n_out = sub.second.get<int>("n_out");
      layer.embedding.push_back(emb);
    }
  }

}
