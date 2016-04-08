#include "lwtnn/NanReplacer.hh"

#include <cmath>  // NAN
#include <limits> // check for NAN

static_assert(std::numeric_limits<double>::has_quiet_NaN,
              "no NaN defined, but we require one");

namespace lwt {
  NanReplacer::NanReplacer(const NanReplacer::ValueMap& defaults, int fg):
    _defaults(defaults),
    _do_nan(fg & rep::nan),
    _do_inf(fg & rep::inf),
    _do_ninf(fg & rep::ninf)
  {
  }

  NanReplacer::ValueMap
  NanReplacer::replace(const NanReplacer::ValueMap& inputs) const {
    // return a new map with the NaN values replaced where possible.
    ValueMap outputs;

    // loop over all inputs
    for (const auto& in: inputs) {
      double val = in.second;

      // figure out if this value should be replaced
      bool is_nan = _do_nan && std::isnan(val);
      bool is_inf = _do_inf && std::isinf(val) && !std::signbit(val);
      bool is_ninf = _do_ninf && std::isinf(val) && std::signbit(val);
      bool is_bad = is_nan || is_inf || is_ninf;

      bool in_defaults = _defaults.count(in.first);
      if (is_bad && in_defaults) {
        outputs[in.first] = _defaults.at(in.first);
      } else {
        outputs[in.first] = val;
      }
    }
    return outputs;
  }
}
