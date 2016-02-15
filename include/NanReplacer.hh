#ifndef NAN_REPLACER_HH
#define NAN_REPLACER_HH

#include <map>
#include <string>

namespace lwt {
  class NanReplacer
  {
  public:
    typedef std::map<std::string, double> ValueMap;
    NanReplacer(const ValueMap& defaults);
    ValueMap replace(const ValueMap& in) const;
  private:
    ValueMap _defaults;
  };
}

#endif
