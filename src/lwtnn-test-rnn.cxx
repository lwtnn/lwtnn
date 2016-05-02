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
  std::cout << "usage: " << name << " <nn config> [<disable>]\n"
            << "\n"
            << "Read in an RNN, feed it data.\n"
            << "With anything for <disable>, just run the random number gen";
}

int main(int argc, char* argv[]) {
  if (argc < 2 || argc > 3) {
    usage(argv[0]);
    exit(1);
  }
  bool run_stack = true;
  if (argc == 3) {
    run_stack = false;
  }
  // Read in the configuration.
  std::string in_file_name(argv[1]);
  std::ifstream in_file(in_file_name);
  auto config = lwt::parse_json(in_file);

  size_t n_inputs = config.inputs.size();
  lwt::RecurrentStack stack(n_inputs, config.layers);
  Eigen::VectorXd sum_outputs = Eigen::VectorXd::Zero(stack.n_outputs());
  Eigen::VectorXd sum_inputs = Eigen::VectorXd::Zero(n_inputs);
  size_t n_loops = 10000;
  std::cout << "running over " << n_loops << " loops" << std::endl;
  for (size_t nnn = 0; nnn < n_loops; nnn++) {
    Eigen::MatrixXd test_pattern = Eigen::MatrixXd::Random(n_inputs, 40);
    for (size_t iii = 0; iii < test_pattern.cols(); iii++) {
      sum_inputs += test_pattern.col(iii);
    }
    if (run_stack) {
      sum_outputs += stack.reduce(test_pattern);
    }
  }
  std::cout << "input sum:\n" << sum_inputs << std::endl;
  std::cout << "output sum:\n" << sum_outputs << std::endl;
  return 0;
}
