#include "lwtnn/LightweightNeuralNetwork.hh"
#include "lwtnn/parse_json.hh"

#include <Eigen/Dense>

#include <iostream>

int main(int argc, char* argv[]) {

  std::vector<double> weights{
    0, 0, 0, 1,
    0, 0, 1, 0,
    0, 1, 0, 0,
    1, 0, 0, 0};
  lwt::LayerConfig layer1{weights};
  lwt::LayerConfig narrowing{
    { 1, 1, 0, 0,
      0, 0, 1, 1} };

  lwt::Stack stack(4, {layer1, narrowing});
  Eigen::VectorXd input(4);
  input << 1, 2, 3, 4;
  std::cout << stack.compute(input) << std::endl;
  return 0;
}
