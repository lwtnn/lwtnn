#include "LWTagger.hh"
#include "parse_json.hh"

#include <Eigen/Dense>

#include <iostream>
#include <string>

int main(int argc, char* argv[]) {
  auto config = lwt::parse_json(std::cin);
  lwt::LWTagger tagger(config.inputs, config.layers, config.outputs);
  lwt::LWTagger::ValueMap input{
    {"in1", 1}, {"in2", 2}, {"in3", 3}, {"in4", 4} };
  auto out = tagger.compute(input);
  for (const auto& op: out) {
    std::cout << op.first << " " << op.second << std::endl;
  }

  return 0;
}
