#ifndef LWT_STREAMERS_HH
#define LWT_STREAMERS_HH

#include <ostream>

namespace lwt {
  struct LayerConfig;
  struct Input;
}

std::ostream& operator<<(std::ostream&, const lwt::LayerConfig&);
std::ostream& operator<<(std::ostream&, const lwt::Input&);

#endif
