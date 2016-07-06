#include "lwtnn/LightweightNeuralNetwork.hh"
#include "lwtnn/Stack.hh"
#include "lwtnn/parse_json.hh"
// #include "lwtnn/NanReplacer.hh"

#include <Eigen/Dense>

#include <iostream>
#include <sstream>
#include <string>
#include <fstream>
#include <cassert>

void usage(const std::string& name) {
  std::cout << "usage: " << name << " <nn config>\n"
            << "\n"
            << "Read in an RNN, feed it data.\n";
}

namespace {
  // 2d ramp function, corners are (1, -1, -1, 1), linear
  // interpolation in the grid between
  double ramp(const lwt::Input& in, size_t x, size_t y,
              size_t n_x, size_t n_y);
}

lwt::VectorMap get_values_vec(const std::vector<lwt::Input>& inputs,
                              size_t n_patterns) {
  lwt::VectorMap out;

  // ramp through the input multiplier
  const size_t total_inputs = inputs.size();
  for (size_t jjj = 0; jjj < n_patterns; jjj++) {
    for (size_t nnn = 0; nnn < total_inputs; nnn++) {
      const auto& input = inputs.at(nnn);
      double ramp_val = ramp(input, nnn, jjj, total_inputs, n_patterns);
      out[input.name].push_back(ramp_val);
    }
  }
  return out;
}

int main(int argc, char* argv[]) {
  if (argc != 2) {
    usage(argv[0]);
    exit(1);
  }
  // Read in the configuration.
  std::string in_file_name(argv[1]);
  std::ifstream in_file(in_file_name);
  auto config = lwt::parse_json(in_file);

  lwt::LightweightRNN rnn(config.inputs, config.layers, config.outputs);
  const auto inputs = get_values_vec(config.inputs, 20);
  auto outs = rnn.reduce(inputs);
  for (const auto& out: outs) {
    std::cout << out.first << " " << out.second << std::endl;
  }
  return 0;
}

namespace {
  // 2d ramp function, see declaration above
  double ramp(const lwt::Input& in, size_t x, size_t y,
              size_t n_x, size_t n_y) {
    assert(x < n_x);
    assert(y < n_y);
    double s_x = 2.0 / (n_x - 1);
    double s_y = 2.0 / (n_y - 1);
    double x_m = ( (n_x == 1) ? 0 : (-1.0 + x * s_x) );
    double y_m = ( (n_y == 1) ? 0 : (-1.0 + y * s_y) );
    return x_m * y_m / in.scale - in.offset;
  }
}
