#include "LWTagger.hh"
#include "parse_json.hh"

#include <Eigen/Dense>

#include <iostream>
#include <string>
#include <fstream>
#include <sstream>

int main(int argc, char* argv[]) {
  if (argc != 2) return 1;
  std::string fname = argv[1];
  std::ifstream infile(fname);
  std::stringstream stst;
  stst << infile.rdbuf();
  auto config = lwt::parse_json(stst);

  return 0;
}
