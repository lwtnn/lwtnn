#include "LightweightNeuralNetwork.hh"
#include "parse_json.hh"

#include <Eigen/Dense>

#include <iostream>
#include <string>
#include <limits>

namespace {
  const double NaN = std::numeric_limits<double>::quiet_NaN();

  // duplicated the inputs, but with NaN replaced with default values
  // may throw a logic_error
  std::map<std::string, double> replace_nan_with_defaults(
    const std::map<std::string, double>& inputs,
    const std::map<std::string, double>& defaults);
}

int main(int argc, char* argv[]) {
  // Read in the configuration. In Athena you should be able to replace
  // std::cin with a stringstream containing the text file
  auto config = lwt::parse_json(std::cin);

  // initialize the tagger from the configuration
  lwt::LightweightNeuralNetwork tagger(
    config.inputs, config.layers, config.outputs);

  // build some dummy inputs and feed them to the tagger
  lwt::LightweightNeuralNetwork::ValueMap input{
    {"in1", 1}, {"in2", 2}, {"in3", NaN}, {"in4", 4} };
  auto cleaned_inputs = replace_nan_with_defaults(input, config.defaults);
  auto out = tagger.compute(cleaned_inputs);

  // look at the outputs
  for (const auto& op: out) {
    std::cout << op.first << " " << op.second << std::endl;
  }

  return 0;
}

namespace {
  std::map<std::string, double> replace_nan_with_defaults(
    const std::map<std::string, double>& inputs,
    const std::map<std::string, double>& defaults) {

    // replace input values, rather than overwriting them
    std::map<std::string, double> outputs;

    // loop over all inputs
    for (const auto& in: inputs) {
      if (std::isnan(in.second)) {
	if (defaults.count(in.first)) {
	  outputs[in.first] = defaults.at(in.first);
	} else {
	  throw std::logic_error(
	    "found nan for " + in.first + " which has no defined default");
	}
      } else {
	outputs[in.first] = in.second;
      }
    }

    return outputs;
  } // end of replace_nan_with_defaults

}
