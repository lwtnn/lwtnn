#include "LightweightNeuralNetwork.hh"
#include "parse_json.hh"
#include "NanReplacer.hh"

#include <Eigen/Dense>

#include <iostream>
#include <string>

int main(int argc, char* argv[]) {
  // Read in the configuration. In Athena you should be able to replace
  // std::cin with a stringstream containing the text file
  auto config = lwt::parse_json(std::cin);

  // initialize the tagger from the configuration
  lwt::LightweightNeuralNetwork tagger(
    config.inputs, config.layers, config.outputs);
  lwt::NanReplacer replacer(config.defaults);

  // build some dummy inputs and feed them to the tagger
  lwt::ValueMap input{
    {"in1", 1}, {"in2", 2}, {"in3", NAN}, {"in4", 4} };
  auto cleaned_inputs = replacer.replace(input);
  auto out = tagger.compute(cleaned_inputs);

  // look at the outputs
  for (const auto& op: out) {
    std::cout << op.first << " " << op.second << std::endl;
  }

  return 0;
}
