#include "test_utilities.hh"

#include <sstream>
#include <cassert>

double ramp(const lwt::Input& in, std::size_t pos, std::size_t n_entries) {
  double step = 2.0 / (n_entries - 1);
  double x = ( (n_entries == 1) ? -1 : (-1 + pos * step) );
  return x / in.scale - in.offset;
}

// 2d ramp function, see declaration above
double ramp(const lwt::Input& in, std::size_t x, std::size_t y,
            std::size_t n_x, std::size_t n_y) {
  assert(x < n_x);
  assert(y < n_y);
  double s_x = 2.0 / (n_x - 1);
  double s_y = 2.0 / (n_y - 1);
  double x_m = ( (n_x == 1) ? 0 : (-1.0 + x * s_x) );
  double y_m = ( (n_y == 1) ? 0 : (-1.0 + y * s_y) );
  return x_m * y_m / in.scale - in.offset;
}


lwt::VectorMap get_values_vec(const std::vector<lwt::Input>& inputs,
                              std::size_t n_patterns) {
  lwt::VectorMap out;

  // ramp through the input multiplier
  const std::size_t total_inputs = inputs.size();
  for (std::size_t nnn = 0; nnn < total_inputs; nnn++) {
    const auto& input = inputs.at(nnn);
    out[input.name] = {};
    for (std::size_t jjj = 0; jjj < n_patterns; jjj++) {
      double ramp_val = ramp(input, nnn, jjj, total_inputs, n_patterns);
      out.at(input.name).push_back(ramp_val);
    }
  }
  return out;
}

std::vector<std::string> parse_line(std::string& line) {
  std::stringstream          line_stream(line);
  std::string                cell;

  std::vector<std::string>   result;
  while(line_stream >> cell) {
    result.push_back(cell);
  }
  return result;
}
