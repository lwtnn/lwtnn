#ifndef STREAMERS_HH
#define STREAMERS_HH

#include <ostream>

namespace lwt {
  struct LayerConfig;
  struct Input;
}

std::ostream& operator<<(std::ostream&, const lwt::LayerConfig&);
std::ostream& operator<<(std::ostream&, const lwt::Input&);

#endif
