/// @file tests/test-nn-streamers.cxx
///
/// Test that the std stream operators work correctly.
///

// System include(s):
#include <iostream>
#include <sstream>
#include <cmath>

/// Class exercising the custom streamers when using a non-std output stream
///
/// This is needed, because clang treats output operators a bit more
/// conservatively than GCC, which only shows up in such a setup.
///
class TestStream : public std::ostringstream {

public:
   /// Inherit the base class's constructors
   using std::ostringstream::ostringstream;

   /// Output operator
   template< class T >
   TestStream& operator<< ( const T& arg ) {
      ( * ( std::ostringstream* ) this ) << arg;
      return *this;
   }

}; // class TestStream

// LWTNN include(s):
#include "lwtnn/NNLayerConfig.hh"
#include "lwtnn/lightweight_network_config.hh"
#include "lwtnn/lightweight_nn_streamers.hh"

int main() {

   // Print a dummy lwt::Input object:
   const lwt::Input dummy1{ "Dummy", 0.0, 1.0 };
   TestStream stream1;
   stream1 << "lwt::Input: " << dummy1;
   std::cout << stream1.str() << std::endl;

   // Print a dumm lwt::LayerConfig object:
   const lwt::LayerConfig dummy2 {
     { 1.0, 2.0 }, { 1.0, 2.0 }, { 1.0, 2.0 },
     {0, lwt::Padding::VALID},
     {lwt::Activation::NONE, NAN}, {lwt::Activation::NONE, NAN},
     {}, {}, {}, lwt::Architecture::NONE};
   TestStream stream2;
   stream2 << "lwt::LayerConfig: " << dummy2;
   std::cout << stream2.str() << std::endl;

   // Return gracefully:
   return 0;
}
