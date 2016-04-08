#ifndef NAN_REPLACER_HH
#define NAN_REPLACER_HH

#include <map>
#include <string>

namespace lwt {
  namespace rep {
    const unsigned nan  = 0x1 << 0;
    const unsigned inf  = 0x1 << 1;
    const unsigned ninf = 0x1 << 2;
    const unsigned all  = nan | inf | ninf;
  }
  class NanReplacer
  {
  public:
    typedef std::map<std::string, double> ValueMap;
    NanReplacer() = default;
    NanReplacer(const ValueMap& defaults, int flags = rep::nan);
    ValueMap replace(const ValueMap& in) const;
  private:
    ValueMap _defaults;
    bool _do_nan;
    bool _do_inf;
    bool _do_ninf;
  };
}

#endif
