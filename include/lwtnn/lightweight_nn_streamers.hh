#ifndef LWT_STREAMERS_HH
#define LWT_STREAMERS_HH

#include <ostream>

namespace lwt {
  struct LayerConfig;
  struct Input;

  std::ostream& operator<<(std::ostream&, const LayerConfig&);
  std::ostream& operator<<(std::ostream&, const Input&);
}

#endif
