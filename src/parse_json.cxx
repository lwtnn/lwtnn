#include "lwtnn/parse_json.hh"

#include <nlohmann/json.hpp>

#include <string>
#include <cmath> // for NAN

namespace {
  using namespace lwt;
  void add_dense_info(LayerConfig& lc, const nlohmann::json& j);
  void add_maxout_info(LayerConfig& lc, const nlohmann::json& j);
  void add_component_info(LayerConfig& lc, const nlohmann::json& j);
  void add_embedding_info(LayerConfig& lc, const nlohmann::json& j);
}


namespace lwt {
  //
  // Define JSON mapping of enum classes
  //
  NLOHMANN_JSON_SERIALIZE_ENUM( Architecture, {
      {Architecture::NONE, "none"},   // need to be first, will also be used for unknown values
      {Architecture::DENSE, "dense"},
      {Architecture::NORMALIZATION, "normalization"},
      {Architecture::HIGHWAY, "highway"},
      {Architecture::MAXOUT, "maxout"},
      {Architecture::LSTM, "lstm"},
      {Architecture::GRU, "gru"},
      {Architecture::SIMPLERNN, "simplernn"},
      {Architecture::EMBEDDING, "embedding"}
    } )
  NLOHMANN_JSON_SERIALIZE_ENUM( NodeConfig::Type, {
      {NodeConfig::Type::NONE, "none"},
      {NodeConfig::Type::FEED_FORWARD, "feed_forward"},
      {NodeConfig::Type::SEQUENCE, "sequence"},
      {NodeConfig::Type::INPUT, "input"},
      {NodeConfig::Type::CONCATENATE, "concatenate"},
      {NodeConfig::Type::TIME_DISTRIBUTED, "time_distributed"},
      {NodeConfig::Type::SUM, "sum"}
    } )
  NLOHMANN_JSON_SERIALIZE_ENUM( Activation, {
      {Activation::NONE, "none"},
      {Activation::LINEAR, "linear"},
      {Activation::SIGMOID, "sigmoid"},
      {Activation::RECTIFIED, "rectified"},
      {Activation::SOFTMAX, "softmax"},
      {Activation::TANH, "tanh"},
      {Activation::HARD_SIGMOID, "hard_sigmoid"},
      {Activation::ELU, "elu"},
      {Activation::LEAKY_RELU, "leakyrelu"},
      {Activation::SWISH, "swish"},
      {Activation::ABS, "abs"},
      // legacy activations. These use UnaryActivationLayer. Just around
      // for benchmarks now.
      {Activation::SIGMOID_LEGACY, "sigmoid_legacy"},
      {Activation::HARD_SIGMOID_LEGACY, "hard_sigmoid_legacy"},
      {Activation::TANH_LEGACY, "tanh_legacy"},
      {Activation::RECTIFIED_LEGACY, "rectified_legacy"}
    } )
  NLOHMANN_JSON_SERIALIZE_ENUM( Component, {
      {Component::I, "i"},
      {Component::O, "o"},
      {Component::C, "c"},
      {Component::F, "f"},
      {Component::Z, "z"},
      {Component::R, "r"},
      {Component::H, "h"},
      {Component::T, "t"},
      {Component::CARRY, "carry"}
    } )

  // Dummy `to_json` methods required even if not used
  void to_json(nlohmann::json&, const ActivationConfig&) {}
  void to_json(nlohmann::json&, const LayerConfig&) {}
  void to_json(nlohmann::json&, const NodeConfig&) {}
  void to_json(nlohmann::json&, const JSONConfig&) {}

  // `from_json` converters for structs that require special treatment
  void from_json(const nlohmann::json& j, ActivationConfig& cfg);
  void from_json(const nlohmann::json& j, LayerConfig& cfg);
  void from_json(const nlohmann::json& j, NodeConfig& cfg);
  void from_json(const nlohmann::json& j, JSONConfig& cfg);

  // "Simple" structs that map directly to JSON
  NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE( Input, name, offset, scale )
  NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE( InputNodeConfig, name, variables ) // defaults, miscellaneous done manually
  NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE( OutputNodeConfig, labels, node_index )
  NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE( EmbeddingConfig, weights, index, n_out )
  NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE( GraphConfig, inputs, input_sequences, nodes, outputs, layers )

  //
  // ActivationConfig
  //
  void from_json(const nlohmann::json& j, ActivationConfig& cfg) {
    // check if this is an "advanced" activation function, in which
    // case it should store the values slightly differently
    if (j.type()!=nlohmann::json::value_t::string) {
      j.at("function").get_to(cfg.function);
      j.at("alpha").get_to(cfg.alpha);
    } else {
      j.get_to(cfg.function);
      cfg.alpha = NAN;
      // special case: kerasfunc2json converter used to pass through
      // the elu activation function. For cases where this has been
      // saved in JSON files we assume an alpha parameter of 1.
      if (cfg.function == Activation::ELU) {
        cfg.alpha = 1.0;
      }
    }
  }

  //
  // LayerConfig
  //
  void from_json(const nlohmann::json& j, LayerConfig& layer) {

    j.at("architecture").get_to(layer.architecture);

    switch(layer.architecture) {
    case Architecture::DENSE:
      add_dense_info(layer, j);
      break;
    case Architecture::NORMALIZATION:
      add_dense_info(layer, j); // re-use dense layer
      break;
    case Architecture::MAXOUT:
      add_maxout_info(layer, j);
      layer.activation.function = Activation::NONE; // FIXME: to make throw_if_not_maxout() happy
      break;
    case Architecture::LSTM:
    case Architecture::GRU:
    case Architecture::SIMPLERNN:
    case Architecture::HIGHWAY:
      add_component_info(layer, j);
      break;
    case Architecture::EMBEDDING:
      add_embedding_info(layer, j);
      break;
    default:
      throw std::logic_error("architecture not implemented");
    }
  }

  //
  // NodeConfig
  //
  void from_json(const nlohmann::json& j, NodeConfig& cfg) {

    j.at("sources").get_to(cfg.sources);
    if (std::any_of(cfg.sources.begin(), cfg.sources.end(), [](int i){return i<0;})) {
      throw std::logic_error("node source number must be positive");
    }
    cfg.type = j.at("type").get<NodeConfig::Type>();

    switch(cfg.type) {
    case NodeConfig::Type::INPUT:
    case NodeConfig::Type::INPUT_SEQUENCE:
      j.at("size").get_to(cfg.index);
      break;
    case NodeConfig::Type::FEED_FORWARD:
    case NodeConfig::Type::SEQUENCE:
    case NodeConfig::Type::TIME_DISTRIBUTED:
      j.at("layer_index").get_to(cfg.index);
      break;
    // Layerless nodes
    case NodeConfig::Type::CONCATENATE:
    case NodeConfig::Type::SUM:
      cfg.index = -1;
      break;
    default:
      throw std::logic_error("unknown node type");
    }
  }

  //
  // JSONConfig
  //
  void from_json(const nlohmann::json& j, JSONConfig& cfg) {
    j.at("inputs").get_to(cfg.inputs);
    j.at("layers").get_to(cfg.layers);
    j.at("outputs").get_to(cfg.outputs);

    // NOTE: at some point we may deprecate this first way of storing
    // default values.
    if (j.contains("defaults")) {
      j["defaults"].get_to(cfg.defaults);
    } else {
      for (const auto& v: j.at("inputs")) {
        if (v.contains("default")) {
          cfg.defaults.emplace(v["name"].get<std::string>(),
                               v["default"].get<double>());
        }
      }
    }

    if (j.contains("miscellaneous")) {
      j["miscellaneous"].get_to(cfg.miscellaneous);
    }
  }

  JSONConfig parse_json(std::istream& json)
  {
    nlohmann::json j;
    json >> j;
    return j.get<JSONConfig>();
  }

  GraphConfig parse_json_graph(std::istream& json) {
    nlohmann::json j;
    json >> j;
    return j.get<GraphConfig>();
  }
}


// Internal helpers
namespace {

  void add_dense_info(LayerConfig& layer, const nlohmann::json& j) {
    j.at("weights").get_to(layer.weights);
    j.at("bias").get_to(layer.bias);

    // this last category is currently only used for LSTM
    if (j.contains("U")) {
      j["U"].get_to(layer.U);
    }

    if (j.contains("activation")) {
      j["activation"].get_to(layer.activation);
    }
  }

  void add_maxout_info(LayerConfig& layer, const nlohmann::json& j) {
    for (const auto& sub: j.at("sublayers")) {
      LayerConfig sublayer;
      add_dense_info(sublayer, sub);
      layer.sublayers.push_back(std::move(sublayer));
    }
  }

  void add_component_info(LayerConfig& layer, const nlohmann::json& j) {
    for (const auto& it : j.at("components").items()) {
      LayerConfig cfg;
      add_dense_info(cfg, it.value());
      Component c;
      from_json(it.key(), c);
      layer.components[c] = cfg;
    }
    j.at("activation").get_to(layer.activation);
    if (j.contains("inner_activation")) {
      j["inner_activation"].get_to(layer.inner_activation);
    }
  }

  void add_embedding_info(LayerConfig& layer, const nlohmann::json& j) {
    for (const auto& sub: j.at("sublayers")) {
      EmbeddingConfig emb;
      sub.at("weights").get_to(emb.weights);
      //emb.index = sub.second.get<int>("index"); FIXME (need an example NN file)
      //emb.n_out = sub.second.get<int>("n_out"); FIXME
      layer.embedding.push_back(std::move(emb));
    }
  }
}
