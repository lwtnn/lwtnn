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

int main(int argc, char* argv[]) {
  if (argc != 2) {
    usage(argv[0]);
    exit(1);
  }
  // Read in the configuration.
  std::string in_file_name(argv[1]);
  std::ifstream in_file(in_file_name);
  std::cout << "reading in NN" << std::endl;
  auto config = lwt::parse_json(in_file);

  size_t n_inputs = config.inputs.size();
  std::cout << "read in NN, constructing recurrent stack..." << std::endl;
  lwt::RecurrentStack stack(n_inputs, config.layers);
  std::cout << "constructed stack, making inputs..." << std::endl;
  Eigen::MatrixXd test_pattern = Eigen::MatrixXd::Random(n_inputs, 10);
  std::cout << "made inputs:\n" << test_pattern << std::endl;
  std::cout << "feeding inputs to RNN" << std::endl;
  Eigen::VectorXd outputs = stack.reduce(test_pattern);
  std::cout << outputs << std::endl;
  return 0;
}
