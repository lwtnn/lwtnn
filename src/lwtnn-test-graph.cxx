// Use example for Graph class

#include "lwtnn/Graph.hh"
#include "lwtnn/parse_json.hh"
#include "lwtnn/NNLayerConfig.hh"

#include <iostream>
#include <fstream>

#include <unistd.h>

namespace {
  lwt::GraphConfig dummy_config();
}

int main(int argc, char* argv[]) {
  // The graph configuration object is returned from the JSON
  // configuration, but it's technically part of the high level
  // interface. Note that we don't need everything in this object to
  // build a Graph.
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

  // Build a dummy source object. A source must implement the ISource
  // interface defined in the Source header. When the graph executes,
  // input nodes will request Eigen::MatrixXd and VectorXd objects
  // from the source object.
  using namespace lwt;
  std::vector<std::size_t> inputs_per_node;
  for (const auto& innode: config.inputs) {
    inputs_per_node.push_back(innode.variables.size());
  }
  std::vector<std::pair<std::size_t, std::size_t> > inputs_per_seq_node;
  for (const auto& innode: config.input_sequences) {
    inputs_per_seq_node.emplace_back(innode.variables.size(),20UL);
  }
  DummySource source(inputs_per_node, inputs_per_seq_node);

  // The Graph is defined by a set of nodes and layers. Again, there's
  // no need to use the high level configuration object here,
  // eveything you need is defined in NNLayerConfig.
  lwt::Graph graph(config.nodes, config.layers);
  // By default `compute` will return the last node which is defined
  // when the graph is constructed. Note that in models with one
  // output this will be the output node, but with multiple outputs
  // this will be ambiguous: you should specify the node index
  // manually as a second argument.
  std::cout << graph.compute(source) << std::endl;
  return 0;
}

namespace {
  lwt::GraphConfig dummy_config() {
    using namespace lwt;
    GraphConfig config;
    std::vector<Input> dummy_inputs{ {"one", 0, 1}, {"two", 0, 1} };
    config.inputs = {{"one", dummy_inputs, {}, {}},
                     {"two", dummy_inputs, {}, {}}};
    typedef NodeConfig::Type Type;
    // First input layer: read from source object at index 0, expect
    // 2d vector.
    config.nodes.push_back({Type::INPUT, {0}, 2});
    // Second input layer: read from source at 1, expect 2d input
    // vector.
    config.nodes.push_back({Type::INPUT, {1}, 2});
    // Concatenate layer: read from layers 0 and 1 above
    config.nodes.push_back({Type::CONCATENATE, {0, 1}, 0});
    // Feed forward layer, read from layer 2, apply transform from
    // layer 0 (defined below.
    config.nodes.push_back({Type::FEED_FORWARD, {2}, 0});
    // Same layer again.
    config.nodes.push_back({Type::FEED_FORWARD, {3}, 0});

    // Simple dense layer, inverts the input vector
    LayerConfig dense;
    dense.weights = {0, 0, 0, 1,  0, 0, 1, 0,  0, 1, 0, 0,  1, 0, 0, 0};
    dense.bias = {0, 0, 0, 0};
    dense.activation.function = Activation::LINEAR;
    dense.architecture = Architecture::DENSE;
    config.layers.push_back(dense);
    return config;
  }
}
