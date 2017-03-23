#include "lwtnn/Graph.hh"

#include <iostream>

int main(int argc, char* argv[]) {
  size_t node_number = 1;
  if (argc > 1) node_number = atoi(argv[1]);
  lwt::DummySource source({2,2});
  lwt::Graph graph;
  std::cout << graph.compute(source) << std::endl;
  return 0;
}
