#include "lwtnn/parse_json.hh"

// this is needed to quiet some warnings from boost
#define BOOST_BIND_GLOBAL_PLACEHOLDERS
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>

#include <cassert>
#include <string>
#include <cmath> // for NAN
#include <set>

namespace {
  using namespace boost::property_tree;
  using namespace lwt;
  LayerConfig get_layer(const ptree::value_type& pt);
  Input get_input(const ptree::value_type& pt);
  InputNodeConfig get_input_node(const ptree::value_type& pt);
  NodeConfig get_node(const ptree::value_type& pt);
  OutputNodeConfig get_output_node(const ptree::value_type& v);
  NodeConfig::Type get_node_type(const std::string&);
  ActivationConfig get_activation(const ptree&);
  Activation get_activation_function(const std::string&);
  Architecture get_architecture(const std::string&);
  void set_defaults(LayerConfig& lc);
  void add_dense_info(LayerConfig& lc, const ptree::value_type& pt);
  void add_maxout_info(LayerConfig& lc, const ptree::value_type& pt);
  void add_component_info(LayerConfig& lc, const ptree::value_type& pt);
  void add_embedding_info(LayerConfig& lc, const ptree::value_type& pt);

  std::map<std::string, double> get_defaults(const ptree& ptree);
}


namespace lwt {

  JSONConfig parse_json(std::istream& json)
  {
    boost::property_tree::ptree pt;
    boost::property_tree::read_json(json, pt);

    JSONConfig cfg;
    for (const auto& v: pt.get_child("inputs")) {
      cfg.inputs.push_back(get_input(v));
    }
    for (const auto& v: pt.get_child("layers")) {
      cfg.layers.push_back(get_layer(v));
    }
    for (const auto& v: pt.get_child("outputs"))
    {
      assert(v.first.empty()); // array elements have no names
      cfg.outputs.push_back(v.second.data());
    }
    cfg.defaults = get_defaults(pt);
    const std::string mname = "miscellaneous";
    if (pt.count(mname)) {
      for (const auto& misc: pt.get_child(mname)) {
        cfg.miscellaneous.emplace(
          misc.first, misc.second.get_value<std::string>());
      }
    }
    return cfg;
  }


  GraphConfig parse_json_graph(std::istream& json) {
    boost::property_tree::ptree pt;
    boost::property_tree::read_json(json, pt);

    GraphConfig cfg;
    for (const auto& v: pt.get_child("inputs")) {
      cfg.inputs.push_back(get_input_node(v));
    }
    for (const auto& v: pt.get_child("input_sequences")) {
      cfg.input_sequences.push_back(get_input_node(v));
    }
    for (const auto& v: pt.get_child("nodes")) {
      cfg.nodes.push_back(get_node(v));
    }
    for (const auto& v: pt.get_child("layers")) {
      cfg.layers.push_back(get_layer(v));
    }
    for (const auto& v: pt.get_child("outputs")) {
      cfg.outputs.emplace(v.first, get_output_node(v));
    }
    return cfg;
  }

}

namespace {

  lwt::Input get_input(const ptree::value_type& v) {
    std::string name = v.second.get<std::string>("name");
    auto offset = v.second.get<double>("offset");
    auto scale = v.second.get<double>("scale");
    return {name, offset, scale};
  }

  lwt::InputNodeConfig get_input_node(const ptree::value_type& v) {
    InputNodeConfig cfg;
    cfg.name = v.second.get<std::string>("name");
    for (const auto& var: v.second.get_child("variables")) {
      cfg.variables.push_back(get_input(var));
      if (var.second.count("default")) {
        std::string name = var.second.get<std::string>("name");
        cfg.defaults.emplace(name, var.second.get<double>("default"));
      }
    }
    return cfg;
  }

  const std::set<NodeConfig::Type> layerless_nodes {
    NodeConfig::Type::CONCATENATE, NodeConfig::Type::SUM, NodeConfig::Type::ADD };
  NodeConfig get_node(const ptree::value_type& v) {
    NodeConfig cfg;

    for (const auto& source: v.second.get_child("sources")) {
      int source_number = source.second.get_value<int>();
      if (source_number < 0) {
        throw std::logic_error("node source number must be positive");
      }
      cfg.sources.push_back(source_number);
    }

    cfg.type = get_node_type(v.second.get<std::string>("type"));
    typedef NodeConfig::Type Type;
    if (cfg.type == Type::INPUT || cfg.type == Type::INPUT_SEQUENCE) {
      cfg.index = v.second.get<int>("size");
    } else if (cfg.type == Type::FEED_FORWARD || cfg.type == Type::SEQUENCE ||
               cfg.type == Type::TIME_DISTRIBUTED) {
      cfg.index = v.second.get<int>("layer_index");
    } else if (layerless_nodes.count(cfg.type)){
      cfg.index = -1;
    } else {
      throw std::logic_error("unknown node type");
    }
    return cfg;
  }

  OutputNodeConfig get_output_node(const ptree::value_type& v) {
    OutputNodeConfig cfg;
    for (const auto& lab: v.second.get_child("labels")) {
      cfg.labels.push_back(lab.second.get_value<std::string>());
    }
    int idx = v.second.get<int>("node_index");
    if (idx < 0) throw std::logic_error("output node index is negative");
    cfg.node_index = idx;
    return cfg;
  }

  NodeConfig::Type get_node_type(const std::string& type) {
    typedef NodeConfig::Type Type;
    if (type == "feed_forward") return Type::FEED_FORWARD;
    if (type == "sequence") return Type::SEQUENCE;
    if (type == "input") return Type::INPUT;
    if (type == "input_sequence") return Type::INPUT_SEQUENCE;
    if (type == "concatenate") return Type::CONCATENATE;
    if (type == "add") return Type::ADD;
    if (type == "time_distributed") return Type::TIME_DISTRIBUTED;
    if (type == "sum") return Type::SUM;
    throw std::logic_error("no node type '" + type + "'");
  }

  LayerConfig get_layer(const ptree::value_type& v) {
    using namespace lwt;
    LayerConfig layer;
    set_defaults(layer);
    Architecture arch = get_architecture(
      v.second.get<std::string>("architecture"));

    if (arch == Architecture::DENSE) {
      add_dense_info(layer, v);
    } else if (arch == Architecture::NORMALIZATION) {
      add_dense_info(layer, v); // re-use dense layer
    } else if (arch == Architecture::MAXOUT) {
      add_maxout_info(layer, v);
    } else if (arch == Architecture::LSTM ||
               arch == Architecture::GRU ||
               arch == Architecture::SIMPLERNN ||
               arch == Architecture::HIGHWAY) {
      add_component_info(layer, v);
    } else if (arch == Architecture::EMBEDDING) {
      add_embedding_info(layer, v);
    } else {
      throw std::logic_error("architecture not implemented");
    }
    layer.architecture = arch;
    return layer;
  }

  lwt::ActivationConfig get_activation(const ptree& v) {
    // check if this is an "advanced" activation function, in which
    // case it should store the values slightly differently
    lwt::ActivationConfig cfg;
    if (v.size() > 0) {
      cfg.function = get_activation_function(
        v.get<std::string>("function"));
      cfg.alpha = v.get<double>("alpha");
    } else {
      cfg.function = get_activation_function(v.data());
      cfg.alpha = NAN;
      // special case: kerasfunc2json converter used to pass through
      // the elu activation function. For cases where this has been
      // saved in JSON files we assume an alpha parameter of 1.
      if (cfg.function == Activation::ELU) {
        cfg.alpha = 1.0;
      }
    }
    return cfg;
  }

  lwt::Activation get_activation_function(const std::string& str) {
    using namespace lwt;
    if (str == "linear") return Activation::LINEAR;
    if (str == "sigmoid") return Activation::SIGMOID;
    if (str == "rectified") return Activation::RECTIFIED;
    if (str == "softmax") return Activation::SOFTMAX;
    if (str == "tanh") return Activation::TANH;
    if (str == "hard_sigmoid") return Activation::HARD_SIGMOID;
    if (str == "elu") return Activation::ELU;
    if (str == "leakyrelu") return Activation::LEAKY_RELU;
    if (str == "swish") return Activation::SWISH;
    if (str == "abs") return Activation::ABS;
    // legacy activations. These use UnaryActivationLayer. Just around
    // for benchmarks now.
    if (str == "sigmoid_legacy") return Activation::SIGMOID_LEGACY;
    if (str == "hard_sigmoid_legacy") return Activation::HARD_SIGMOID_LEGACY;
    if (str == "tanh_legacy") return Activation::TANH_LEGACY;
    if (str == "rectified_legacy") return Activation::RECTIFIED_LEGACY;
    throw std::logic_error("activation function " + str + " not recognized");
    return Activation::LINEAR;
  }


  lwt::Architecture get_architecture(const std::string& str) {
    using namespace lwt;
    if (str == "dense") return Architecture::DENSE;
    if (str == "normalization") return Architecture::NORMALIZATION;
    if (str == "highway") return Architecture::HIGHWAY;
    if (str == "maxout") return Architecture::MAXOUT;
    if (str == "lstm") return Architecture::LSTM;
    if (str == "gru") return Architecture::GRU;
    if (str == "simplernn") return Architecture::SIMPLERNN;
    if (str == "embedding") return Architecture::EMBEDDING;
    throw std::logic_error("architecture " + str + " not recognized");
  }

  void set_defaults(LayerConfig& layer) {
    layer.activation.function = Activation::NONE;
    layer.inner_activation.function = Activation::NONE;
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
      layer.activation = get_activation(v.second.get_child("activation"));
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


  const std::map<std::string, lwt::Component> component_map {
    {"i", Component::I},
    {"o", Component::O},
    {"c", Component::C},
    {"f", Component::F},
    {"z", Component::Z},
    {"r", Component::R},
    {"h", Component::H},
    {"t", Component::T},
    {"carry", Component::CARRY}
  };

  void add_component_info(LayerConfig& layer, const ptree::value_type& v) {
    using namespace lwt;
    for (const auto& comp: v.second.get_child("components")) {
      LayerConfig cfg;
      set_defaults(cfg);
      add_dense_info(cfg, comp);
      layer.components[component_map.at(comp.first)] = cfg;
    }
    layer.activation = get_activation(v.second.get_child("activation"));
    if (v.second.count("inner_activation") != 0) {
      layer.inner_activation = get_activation(
        v.second.get_child("inner_activation"));
    }
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

  std::map<std::string, double> get_defaults(const ptree& pt) {
    const std::string dname = "defaults";
    std::map<std::string, double> defaults;
    // NOTE: at some point we may deprecate this first way of storing
    // default values.
    if (pt.count(dname)) {
      for (const auto& def: pt.get_child(dname)) {
        defaults.emplace(def.first, def.second.get_value<double>());
      }
    } else {
      const std::string dkey = "default";
      for (const auto& v: pt.get_child("inputs")) {
        if (v.second.count(dkey)) {
          std::string key = v.second.get<std::string>("name");
          defaults.emplace(key, v.second.get<double>(dkey));
        }
      }
    }
    return defaults;
  }
}
