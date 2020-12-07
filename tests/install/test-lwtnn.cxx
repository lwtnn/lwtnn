// basic script to test cmake configuration and linking

#include "lwtnn/LightweightGraph.hh"
#include "lwtnn/generic/FastGraph.hh"
#include "lwtnn/parse_json.hh"
#include <string>
#include <fstream>
#include <iostream>

int main(int narg, char* argv[]) {
  std::string input_file_path = "bludo";
  if (narg == 1) {
    input_file_path = argv[1];
  }
  std::cout << "testing with " << input_file_path << std::endl;
  std::ifstream in_file(input_file_path);
  auto config = lwt::parse_json_graph(in_file);

  lwt::LightweightGraph graph(config);
  lwt::InputOrder order;
  lwt::generic::FastGraph<float> fast_graph(config, order);
  return 0;
}
