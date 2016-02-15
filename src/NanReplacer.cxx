#include "NanReplacer.hh"

#include <cmath>  // NAN
#include <limits> // check for NAN

static_assert(std::numeric_limits<double>::has_quiet_NaN,
              "no NaN defined, but we require one");

namespace lwt {
  NanReplacer::NanReplacer(const NanReplacer::ValueMap& defaults, int fg):
    m_defaults(defaults),
    m_do_nan(fg & rep::nan),
    m_do_inf(fg & rep::inf),
    m_do_ninf(fg & rep::ninf)
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
      bool is_nan = m_do_nan && std::isnan(val);
      bool is_inf = m_do_inf && std::isinf(val) && !std::signbit(val);
      bool is_ninf = m_do_ninf && std::isinf(val) && std::signbit(val);
      bool is_bad = is_nan || is_inf || is_ninf;

      bool in_defaults = m_defaults.count(in.first);
      if (is_bad && in_defaults) {
        outputs[in.first] = m_defaults.at(in.first);
      } else {
        outputs[in.first] = val;
      }
    }
    return outputs;
  }
}
