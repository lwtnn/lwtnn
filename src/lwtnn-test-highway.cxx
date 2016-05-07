#include "lwtnn/LightweightNeuralNetwork.hh"
#include "lwtnn/parse_json.hh"
#include "lwtnn/NanReplacer.hh"

#include <Eigen/Dense>

#include <iostream>
#include <sstream>
#include <string>
#include <fstream>


using Eigen::VectorXd;
using Eigen::MatrixXd;

int main(int argc, char* argv[]) {
	MatrixXd W(5, 5);
	W << 3, 4, 2, 4, 6,
	    3, 4, 2, 4, 6,
	    3, 4, 2, 4, 6,
	    3, 4, 2, 4, 6,
	    3, 4, 2, 4, 6;

	VectorXd b(5);

	b << 4, 3, 1, 2, 5;

	VectorXd x(5);

	x << 10.32, 2.32, 1.32, 5.3, 0.01;

	lwt::DenseLayer layer1(W, b, lwt::Activation::RECTIFIED);
	std::cout << layer1.compute(x) << std::endl;

	lwt::HighwayLayer hw(W, b, 0.17 * W, 3 * b, lwt::Activation::RECTIFIED);
	std::cout << hw.compute(x) << std::endl;
}	