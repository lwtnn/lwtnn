#include "LightweightNeuralNetwork.hh"
#include "parse_json.hh"

#include <Eigen/Dense>

#include <iostream>
#include <string>
#include <limits> // check for NAN
#include <cmath>  // NAN

namespace {
  static_assert(std::numeric_limits<double>::has_quiet_NaN,
		"no NaN defined, but we require one");

  // copy the inputs, but with NaN replaced with default values. Don't
  // do anything with NaN values with no defined default.
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
    {"in1", 1}, {"in2", 2}, {"in3", NAN}, {"in4", 4} };
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

    // return a new map with the NaN values replaced where possible.
    std::map<std::string, double> outputs;

    // loop over all inputs
    for (const auto& in: inputs) {
      if (std::isnan(in.second) && defaults.count(in.first)) {
	outputs[in.first] = defaults.at(in.first);
      } else {
	outputs[in.first] = in.second;
      }
    }

    return outputs;
  } // end of replace_nan_with_defaults

}
