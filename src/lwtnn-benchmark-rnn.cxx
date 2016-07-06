#include "lwtnn/LightweightNeuralNetwork.hh"
#include "lwtnn/Stack.hh"
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

std::vector<lwt::ValueMap> get_values(
  const std::vector<lwt::Input>& inputs) {
  Eigen::MatrixXd test_pattern = Eigen::MatrixXd::Random(inputs.size(), 40);
  std::vector<lwt::ValueMap> out;
  const auto n_cols = static_cast<size_t>(test_pattern.cols());
  for (size_t iii = 0; iii < n_cols; iii++) {
    lwt::ValueMap vals;
    for (size_t jjj = 0; jjj < inputs.size(); jjj++) {
      vals[inputs.at(jjj).name] = test_pattern(jjj, iii);
    }
    out.push_back(vals);
  }
  return out;
}

lwt::VectorMap get_values_vec(const std::vector<lwt::Input>& inputs) {
  Eigen::MatrixXd test_pattern = Eigen::MatrixXd::Random(inputs.size(), 40);
  lwt::VectorMap out;
  for (size_t in_num = 0; in_num < inputs.size(); in_num++) {
    std::vector<double> ins;
    const auto n_cols = static_cast<size_t>(test_pattern.cols());
    for (size_t iii = 0; iii < n_cols; iii++) {
      ins.push_back(test_pattern(in_num, iii));
    }
    out[inputs.at(in_num).name] = std::move(ins);
  }
  return out;
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

  if (!config.miscellaneous.count("sort_order")) {
    std::cout << "no sort order given!" << std::endl;
  } else {
    std::cout << "sort order: " << config.miscellaneous.at("sort_order")
              << std::endl;
  }

  size_t n_inputs = config.inputs.size();
  lwt::RecurrentStack stack(n_inputs, config.layers);
  lwt::LightweightRNN rnn(config.inputs, config.layers, config.outputs);
  Eigen::VectorXd sum_outputs = Eigen::VectorXd::Zero(stack.n_outputs());
  size_t n_loops = 1;
  std::cout << "running over " << n_loops << " loops" << std::endl;
  std::cout << "running " << (run_stack ? "fast": "slow") << std::endl;
  for (size_t nnn = 0; nnn < n_loops; nnn++) {
    if (run_stack) {
      Eigen::MatrixXd test_pattern = Eigen::MatrixXd::Random(n_inputs, 2);
      std::cout << test_pattern << std::endl;
      sum_outputs += stack.reduce(test_pattern);
    } else {
      const auto inputs = get_values_vec(config.inputs);
      auto out = rnn.reduce(inputs);
      for (size_t iii = 0; iii < config.outputs.size(); iii++) {
        sum_outputs(iii) += out.at(config.outputs.at(iii));
      }
    }
  }
  std::cout << "output sum:\n" << sum_outputs << std::endl;
  return 0;
}
