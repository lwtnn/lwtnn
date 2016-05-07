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

	// -- Test Dense:
	lwt::DenseLayer layer1(W, b, lwt::Activation::RECTIFIED);
	std::cout << "Predictions from lwtnn Dense:" << std::endl;
	std::cout << layer1.compute(x) << std::endl;

	// -- Test Highway:
	lwt::HighwayLayer hw(W, b, 0.17 * W, -3 * b, lwt::Activation::RECTIFIED);
	std::cout << "Predictions from lwtnn Highway:" << std::endl;
	std::cout << hw.compute(x) << std::endl;
}	

// These predictions are to be compared to the ones obtained with Keras and found here:
// http://nbviewer.jupyter.org/gist/mickypaganini/e997d21682a24e6e8c474af25760afa0

