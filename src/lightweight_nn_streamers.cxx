#include "lwtnn/lightweight_nn_streamers.hh"
#include "lwtnn/NNLayerConfig.hh"

#include <vector>
#include <string>

namespace {
  std::ostream& operator<<(std::ostream& out,
			   const std::vector<double>& vec) {
    size_t nentry = vec.size();
    for (size_t iii = 0; iii < nentry; iii++) {
      out << vec.at(iii);
      if (iii < (nentry - 1)) out << ", ";
    }
    return out;
  }
  std::string activation_as_string(lwt::Activation act) {
    using namespace lwt;
    switch (act) {
    case Activation::NONE: return "none";
    case Activation::LINEAR: return "linear";
    case Activation::SIGMOID: return "sigmoid";
    case Activation::RECTIFIED: return "rectified";
    case Activation::SOFTMAX: return "softmax";
    case Activation::TANH: return "tanh";
    case Activation::HARD_SIGMOID: return "hard_sigmoid";
    }
  }
  std::string arch_as_string(lwt::Architecture arch) {
    using namespace lwt;
    switch (arch) {
    case Architecture::NONE: return "none";
    case Architecture::DENSE: return "dense";
    case Architecture::MAXOUT: return "maxout";
    case Architecture::LSTM: return "lstm";
    case Architecture::EMBEDDING: return "embedding";
    }
  }
  std::ostream& operator<<(std::ostream& out, lwt::Architecture ach) {
    out << arch_as_string(ach);
    return out;
  }
  std::string component_as_string(lwt::Component comp) {
    using namespace lwt;
    switch (comp) {
    case Component::I: return "i";
    case Component::O: return "o";
    case Component::C: return "c";
    case Component::F: return "f";
    }
  }
}


std::ostream& operator<<(std::ostream& out, const lwt::LayerConfig& cfg){
  using namespace lwt;
  out << "architecture: " << cfg.architecture << "\n";
  out << "activation: " << activation_as_string(cfg.activation) << "\n";
  if (cfg.inner_activation != Activation::NONE) {
    out << "inner activation: "
        << activation_as_string(cfg.inner_activation) << "\n";
  }
  out << "weights: [" << cfg.weights << "]\n";
  out << "bias: [" << cfg.bias << "]\n";
  if (cfg.U.size() > 0) {
    out << "U: [" << cfg.U << "]\n";
  }
  for (const auto& emb: cfg.embedding) {
    out << "embedding - index: " << emb.index << " n_out: " << emb.n_out
        << "\n";
  }
  if (cfg.sublayers.size() > 0) {
    out << "[\n";
    for (const auto& sub: cfg.sublayers) {
      out << "  " << sub << "\n";
    }
    out << "]\n";
  }
  if (cfg.components.size() > 0) {
    out << "{\n";
    for (const auto& comp: cfg.components) {
      out << " - component: " << component_as_string(comp.first) << " -\n";
      out << comp.second << "\n";
    }
    out << "}\n";
  }
  return out;
}
std::ostream& operator<<(std::ostream& out, const lwt::Input& input) {
  out << input.name << " -- offset: " << input.offset << " scale: "
      << input.scale;
  return out;
}
