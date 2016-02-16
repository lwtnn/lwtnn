#include "lightweight_nn_streamers.hh"
#include "NNLayerConfig.hh"

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
    case Activation::LINEAR: return "linear";
    case Activation::SIGMOID: return "sigmoid";
    case Activation::RECTIFIED: return "rectified";
    case Activation::SOFTMAX: return "softmax";
    }
  }
}


std::ostream& operator<<(std::ostream& out, const lwt::LayerConfig& cfg){
  out << "activation: " << activation_as_string(cfg.activation) << " ";
  out << "weights: [" << cfg.weights << "] ";
  out << "bias: [" << cfg.bias << "]";
  return out;
}
std::ostream& operator<<(std::ostream& out, const lwt::Input& input) {
  out << input.name << " -- offset: " << input.offset << " scale: "
      << input.scale;
  return out;
}
