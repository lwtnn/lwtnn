#include "lwtnn/LightweightNeuralNetwork.hh"
#include "lwtnn/LightweightRNN.hh"
#include "lwtnn/parse_json.hh"
// #include "lwtnn/NanReplacer.hh"

#include <Eigen/Dense>

#include <iostream>
#include <sstream>
#include <string>
#include <fstream>

void usage(const std::string& name) {
  std::cout << "usage: " << name << " <nn config>\n"
            << "\n"
            << "Read in an RNN, feed it data.\n";
}

namespace {
  double ramp(const lwt::Input& in, size_t pos, size_t n_entries);
}

lwt::VectorMap get_values_vec(const std::vector<lwt::Input>& inputs,
                              size_t n_patterns) {
  lwt::VectorMap out;

  // ramp through the input multiplier
  const double step = 2.0 / (n_patterns - 1);
  const size_t total_inputs = inputs.size();
  for (size_t jjj = 0; jjj < n_patterns; jjj++) {
    const double vmult = (-1 + jjj * step);
    for (size_t nnn = 0; nnn < total_inputs; nnn++) {
      const auto& input = inputs.at(nnn);
      double ramp_val = ramp(input, nnn, total_inputs);
      // record the product of the two ramps
      double final_val = vmult * ramp_val;
      out[input.name].push_back(final_val);
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
  double ramp(const lwt::Input& in, size_t pos, size_t n_entries) {
    double step = 2.0 / (n_entries - 1);
    double x = ( (n_entries == 1) ? 0 : (-1 + pos * step) );
    return x / in.scale - in.offset;
  }
}
