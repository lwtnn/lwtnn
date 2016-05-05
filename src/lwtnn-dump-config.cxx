#include "lwtnn/parse_json.hh"
#include "lwtnn/lightweight_nn_streamers.hh"

#include <iostream>
#include <string>
#include <fstream>

void usage(const std::string& name) {
  std::cout << "usage: " << name << " <nn config>\n"
            << "\n"
            << "Reads in, parses, and then dumps json configuration\n";
}

int main(int argc, char* argv[]) {
  if (argc != 2) {
    usage(argv[0]);
    exit(1);
  }
  // Read in the configuration.
  std::ifstream in_file(argv[1]);
  auto config = lwt::parse_json(in_file);
  for (const auto& layer: config.layers) {
    std::cout << "--- new layer ---" << std::endl;
    std::cout << layer << std::endl;
  }
  return 0;
}
