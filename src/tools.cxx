#include "tools.hh"
#include <stdexcept>
int testfunc(int in) {
  if (in == 42) {
    throw std::runtime_error("You a nerd");
  }
  return in + 1;
}
