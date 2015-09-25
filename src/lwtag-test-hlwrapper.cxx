#include "LightweightNeuralNetwork.hh"

#include <Eigen/Dense>

#include <iostream>

int main(int argc, char* argv[]) {

  std::vector<double> weights{
    0, 0, 0, 1,
    0, 0, 1, 0,
    0, 1, 0, 0,
    1, 0, 0, 0};
  lwt::LayerConfig layer1{weights};
  std::vector<lwt::Input> input_conf{
    {"1", 0, 1}, {"2", 0, 1}, {"3", 0, 1}, {"4", 0, 1}};
  std::vector<std::string> outputs{"1", "2", "3", "4"};

  lwt::LightweightNeuralNetwork tagger(input_conf, {layer1}, outputs);
  lwt::LightweightNeuralNetwork::ValueMap input{ {"1", 1}, {"2", 2}, {"3", 3}, {"4", 4} };
  auto out = tagger.compute(input);
  for (const auto& op: out) {
    std::cout << op.first << " " << op.second << std::endl;
  }
  return 0;
}
