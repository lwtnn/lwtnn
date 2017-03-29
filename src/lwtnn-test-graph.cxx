#include "lwtnn/Graph.hh"
#include "lwtnn/parse_json.hh"
#include "lwtnn/NNLayerConfig.hh"

#include <iostream>
#include <fstream>

#include <unistd.h>

lwt::GraphConfig dummy_config() {
  using namespace lwt;
  GraphConfig config;
  std::vector<Input> dummy_inputs{ {"one", 0, 1}, {"two", 0, 1} };
  config.inputs = {{"one", dummy_inputs}, {"two", dummy_inputs}};
  typedef NodeConfig::Type Type;
  config.nodes.push_back({Type::INPUT, {0}, 2});
  config.nodes.push_back({Type::INPUT, {1}, 2});
  config.nodes.push_back({Type::CONCATENATE, {0, 1}});
  config.nodes.push_back({Type::FEED_FORWARD, {2}, 0});
  config.nodes.push_back({Type::FEED_FORWARD, {3}, 0});

  LayerConfig dense {
    {0, 0, 0, 1,  0, 0, 1, 0,  0, 1, 0, 0,  1, 0, 0, 0}};
  dense.activation = Activation::LINEAR;
  dense.architecture = Architecture::DENSE;
  config.layers.push_back(dense);
  return config;
}

int main(int argc, char* argv[]) {
  lwt::GraphConfig config;
  bool is_pipe = !isatty(fileno(stdin));
  if (is_pipe) {
    config = lwt::parse_json_graph(std::cin);
  } else if (argc == 2) {
    std::string in_file_name(argv[1]);
    std::ifstream in_file(in_file_name);
    config = lwt::parse_json_graph(in_file);
  } else {
    config = dummy_config();
  }
  int node_number = -1;
  // if (argc > 1) node_number = atoi(argv[1]);
  using namespace lwt;
  std::vector<size_t> inputs_per_node;
  for (const auto& innode: config.inputs) {
    inputs_per_node.push_back(innode.variables.size());
  }
  std::vector<std::pair<size_t, size_t> > inputs_per_seq_node;
  for (const auto& innode: config.input_sequences) {
    inputs_per_seq_node.emplace_back(innode.variables.size(),10UL);
  }
  DummySource source(inputs_per_node, inputs_per_seq_node);

  lwt::Graph graph(config.nodes, config.layers);
  if (node_number < 0) {
    std::cout << graph.compute(source) << std::endl;
  } else {
    std::cout << graph.compute(source, node_number) << std::endl;
  }
  return 0;
}
