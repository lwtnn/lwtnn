#include "LWTagger.hh"
#include "parse_json.hh"

#include <Eigen/Dense>

#include <iostream>
#include <string>

int main(int argc, char* argv[]) {
  // Read in the configuration. In Athena you should be able to replace
  // std::cin with a stringstream containing the text file
  auto config = lwt::parse_json(std::cin);

  // initialize the tagger from the configuration
  lwt::LWTagger tagger(config.inputs, config.layers, config.outputs);

  // build some dummy inputs and feed them to the tagger
  lwt::LWTagger::ValueMap input{
    {"in1", 1}, {"in2", 2}, {"in3", 3}, {"in4", 4} };
  auto out = tagger.compute(input);

  // look at the outputs
  for (const auto& op: out) {
    std::cout << op.first << " " << op.second << std::endl;
  }

  return 0;
}
