#include "lwtnn/Graph.hh"
#include "lwtnn/NNLayerConfig.hh"

#include <iostream>

int main(int argc, char* argv[]) {
  size_t node_number = 1;
  if (argc > 1) node_number = atoi(argv[1]);
  using namespace lwt;
  typedef NodeConfig::Type Type;
  DummySource source({2,2});
  std::vector<NodeConfig> nodes;
  nodes.push_back({Type::INPUT, {0}, 2});
  nodes.push_back({Type::INPUT, {1}, 2});
  nodes.push_back({Type::CONCATENATE, {0, 1}});
  nodes.push_back({Type::FEED_FORWARD, {2}, 0});
  nodes.push_back({Type::FEED_FORWARD, {3}, 0});

  LayerConfig dense {
    {0, 0, 0, 1,  0, 0, 1, 0,  0, 1, 0, 0,  1, 0, 0, 0}};
  dense.activation = Activation::LINEAR;
  dense.architecture = Architecture::DENSE;

  lwt::Graph graph(nodes, {dense});
  std::cout << graph.compute(source, node_number) << std::endl;
  return 0;
}
