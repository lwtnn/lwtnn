#include "NanReplacer.hh"

#include <cmath>  // NAN
#include <limits> // check for NAN

static_assert(std::numeric_limits<double>::has_quiet_NaN,
              "no NaN defined, but we require one");

namespace lwt {
  NanReplacer::NanReplacer(const NanReplacer::ValueMap& defaults):
    _defaults(defaults)
  {}

  NanReplacer::ValueMap
  NanReplacer::replace(const NanReplacer::ValueMap& inputs) const {
    // return a new map with the NaN values replaced where possible.
    ValueMap outputs;

    // loop over all inputs
    for (const auto& in: inputs) {
      if (std::isnan(in.second) && _defaults.count(in.first)) {
        outputs[in.first] = _defaults.at(in.first);
      } else {
        outputs[in.first] = in.second;
      }
    }
    return outputs;
  }
}
