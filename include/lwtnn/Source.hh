#ifndef LWTNN_SOURCE_HH
#define LWTNN_SOURCE_HH

#include "lwtnn/generic/Source.hh"

namespace lwt {

  using ISource = generic::ISource<double>;
  using VectorSource = generic::VectorSource<double>;
  using DummySource = generic::DummySource<double>;

}

#endif
