#include "LightweightNeuralNetwork.hh"
#include "parse_json.hh"
#include "NanReplacer.hh"

#include <Eigen/Dense>

#include <iostream>
#include <sstream>
#include <string>
#include <fstream>


using Eigen::VectorXd;
using Eigen::MatrixXd;

int main(int argc, char* argv[]) {
  MatrixXd W(4, 5);
  W << 3, 4, 2, 4, 6,
       3, 4, 2, 4, 6,
       3, 4, 2, 4, 6,
       3, 4, 2, 4, 6;

  VectorXd b(4);

  b << 4, 3, 1, 2;


  VectorXd x(5);

  x << 10.32, 2.32, 1.32, 5.3, 0.01;

  // lwt::DenseLayer layer1(W, b, new lwt::RectifiedLayer());

  // lwt::HighwayLayer hw(carry, transform);
  lwt::HighwayLayer hw(W, b, 0.17 * W, 3 * b, lwt::RectifiedLayer());

  // std::cout << layer1.compute(x) << std::endl;

  

  




}
