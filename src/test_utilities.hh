#ifndef TEST_UTILITIES_HH
#define TEST_UTILITIES_HH

//////////////////////////
// lwtnn test utilities //
//////////////////////////
//
// This is for a few shared functions for simple unit tests.
//

#include "lwtnn/lightweight_network_config.hh"

// ramp function so that the inputs _after_ normalization fall on the
// [-1,1] range, i.e. `np.linspace(-1, 1, n_entries)`
double ramp(const lwt::Input& in, std::size_t pos, std::size_t n_entries);

// 2d ramp function, corners in (x, y) are (-1, -1), (1, 1), linear
// interpolation in the grid between. This can be reproduced in numpy
// with
//
// np.linspace(-1, 1, 20)[:,None] * np.linspace(-1, 1, n_features)[None,:]
//
double ramp(const lwt::Input& in, std::size_t x, std::size_t y,
            std::size_t n_x, std::size_t n_y);

lwt::VectorMap get_values_vec(const std::vector<lwt::Input>& inputs,
                                std::size_t n_patterns);

std::vector<std::string> parse_line(std::string& line);

#endif
