#include "lwtnn/Graph.hh"
#include "lwtnn/Exceptions.hh"
#include "lwtnn/Stack.hh"

#include <set>

namespace lwt {

  // Sources
  VectorSource::VectorSource(std::vector<VectorXd>&& vv,
                             std::vector<MatrixXd>&& mm):
    m_inputs(std::move(vv)),
    m_matrix_inputs(std::move(mm))
  {
  }
  VectorXd VectorSource::at(size_t index) const {
    if (index >= m_inputs.size()) {
      throw NNEvaluationException(
        "VectorSource: no source vector defined at " + std::to_string(index));
    }
    return m_inputs.at(index);
  }
  MatrixXd VectorSource::matrix_at(size_t index) const {
    if (index >= m_matrix_inputs.size()) {
      throw NNEvaluationException(
        "VectorSource: no source matrix defined at " + std::to_string(index));
    }
    return m_matrix_inputs.at(index);
  }

  DummySource::DummySource(const std::vector<size_t>& input_sizes,
                           const std::vector<std::pair<size_t,size_t> >& ma):
    m_sizes(input_sizes),
    m_matrix_sizes(ma)
  {
  }
  VectorXd DummySource::at(size_t index) const {
    if (index >= m_sizes.size()) {
      throw NNEvaluationException(
        "Dummy Source: no size defined at " + std::to_string(index));
    }
    size_t n_entries = m_sizes.at(index);
    VectorXd vec(n_entries);
    for (size_t iii = 0; iii < n_entries; iii++) {
      vec(iii) = iii;
    }
    return vec;
  }
  MatrixXd DummySource::matrix_at(size_t index) const {
    if (index >= m_sizes.size()) {
      throw NNEvaluationException(
        "Dummy Source: no size defined at " + std::to_string(index));
    }
    size_t n_rows = m_matrix_sizes.at(index).first;
    size_t n_cols = m_matrix_sizes.at(index).second;
    MatrixXd mat(n_rows, n_cols);
    for (size_t iii = 0; iii < n_rows; iii++) {
      for (size_t jjj = 0; jjj < n_cols; jjj++) {
        mat(iii, jjj) = jjj + n_cols * iii;
      }
    }
    return mat;
  }


  // Nodes
  InputNode::InputNode(size_t index, size_t n_outputs):
    m_index(index),
    m_n_outputs(n_outputs)
  {
  }
  VectorXd InputNode::compute(const ISource& source) const {
    VectorXd output = source.at(m_index);
    assert(output.rows() > 0);
    if (static_cast<size_t>(output.rows()) != m_n_outputs) {
      std::string len = std::to_string(output.rows());
      std::string found = std::to_string(m_n_outputs);
      throw NNEvaluationException(
        "Found vector of length " + len + ", expected " + found);
    }
    return output;
  }
  size_t InputNode::n_outputs() const {
    return m_n_outputs;
  }

  FeedForwardNode::FeedForwardNode(const Stack* stack, const INode* source):
    m_stack(stack),
    m_source(source)
  {
  }
  VectorXd FeedForwardNode::compute(const ISource& source) const {
    return m_stack->compute(m_source->compute(source));
  }
  size_t FeedForwardNode::n_outputs() const {
    return m_stack->n_outputs();
  }

  ConcatenateNode::ConcatenateNode(const std::vector<const INode*>& sources):
    m_sources(sources),
    m_n_outputs(0)
  {
    for (const auto source: sources) {
      m_n_outputs += source->n_outputs();
    }
  }
  VectorXd ConcatenateNode::compute(const ISource& source) const {
    VectorXd output(m_n_outputs);
    size_t offset = 0;
    for (const auto node: m_sources) {
      VectorXd input = node->compute(source);
      size_t n_elements = input.rows();
      assert(n_elements == node->n_outputs());
      output.segment(offset, n_elements) = input;
      offset += n_elements;
    }
    assert(offset = m_n_outputs);
    return output;
  }
  size_t ConcatenateNode::n_outputs() const {
    return m_n_outputs;
  }

  // Sequence nodes
  InputSequenceNode::InputSequenceNode(size_t index, size_t n_outputs):
    m_index(index),
    m_n_outputs(n_outputs)
  {
  }
  MatrixXd InputSequenceNode::scan(const ISource& source) const {
    MatrixXd output = source.matrix_at(m_index);
    assert(output.rows() > 0);
    if (output.cols() == 0) {
      throw NNEvaluationException("zero length input sequence");
    }
    if (static_cast<size_t>(output.rows()) != m_n_outputs) {
      std::string len = std::to_string(output.rows());
      std::string found = std::to_string(m_n_outputs);
      throw NNEvaluationException(
        "Found vector of length " + len + ", expected " + found);
    }
    return output;
  }
  size_t InputSequenceNode::n_outputs() const {
    return m_n_outputs;
  }

  SequenceNode::SequenceNode(const RecurrentStack* stack,
                             const ISequenceNode* source) :
    m_stack(stack),
    m_source(source)
  {
  }
  MatrixXd SequenceNode::scan(const ISource& source) const {
    return m_stack->scan(m_source->scan(source));
  }
  VectorXd SequenceNode::compute(const ISource& src) const {
    MatrixXd mat = scan(src);
    size_t n_cols = mat.cols();
    return mat.col(n_cols - 1);
  }
  size_t SequenceNode::n_outputs() const {
    return m_stack->n_outputs();
  }
}

namespace {
  using namespace lwt;
  void throw_cfg(std::string msg, size_t index) {
    throw NNConfigurationException(msg + " " + std::to_string(index));
  }
  // NOTE: you own this pointer!
  INode* get_feedforward_node(const NodeConfig& node,
                              const std::vector<LayerConfig>& layers,
                              const std::map<size_t, INode*>& node_map,
                              std::map<size_t, Stack*>& stack_map,
                              std::vector<Stack*>& m_stacks) {

    size_t n_source = node.sources.size();
    if (n_source != 1) throw_cfg("need one source, found", n_source);
    INode* source = node_map.at(node.sources.at(0));

    int layer_n = node.index;
    if (layer_n < 0) throw_cfg("negative layer number", layer_n);
    if (static_cast<size_t>(layer_n) >= layers.size()) {
      throw_cfg("no layer number", layer_n);
    }
    if (!stack_map.count(layer_n)) {
      m_stacks.push_back(
        new Stack(source->n_outputs(), {layers.at(layer_n)}));
      stack_map[layer_n] = m_stacks.back();
    }
    return new FeedForwardNode(stack_map.at(layer_n), source);
  }

  struct GraphMaps
  {
    std::map<size_t, INode*> node;
    std::map<size_t, Stack*> stack;
  };

  void build_node(size_t iii,
                  const std::vector<NodeConfig>& nodes,
                  const std::vector<LayerConfig>& layers,
                  std::vector<INode*>& m_nodes,
                  std::vector<Stack*>& m_stacks,
                  GraphMaps& maps,
                  std::set<size_t> cycle_check = {}) {
    if (maps.node.count(iii)) return;
    if (iii >= nodes.size()) throw_cfg("no node index", iii);

    const NodeConfig& node = nodes.at(iii);

    // if it's an input, build and return
    if (node.type == NodeConfig::Type::INPUT) {
      size_t n_inputs = node.sources.size();
      if (n_inputs != 1) throw_cfg(
        "input node needs need one source, got", n_inputs);
      if (node.index < 0) throw_cfg(
        "input node needs positive index, got", node.index);
      m_nodes.push_back(new InputNode(node.sources.at(0), node.index));
      maps.node[iii] = m_nodes.back();
      return;
    }

    // otherwise build all the inputs first
    if (cycle_check.count(iii)) {
      throw NNConfigurationException("found cycle in graph");
    }
    cycle_check.insert(iii);
    for (size_t source_node: node.sources) {
      build_node(source_node, nodes, layers,
                 m_nodes, m_stacks, maps, cycle_check);
    }

    // build feed forward layer
    if (node.type == NodeConfig::Type::FEED_FORWARD) {
      m_nodes.push_back(
        get_feedforward_node(node, layers, maps.node, maps.stack, m_stacks));
      maps.node[iii] = m_nodes.back();
      return;
    }

    // build concatenate layer
    if (node.type == NodeConfig::Type::CONCATENATE) {
      std::vector<const INode*> in_nodes;
      for (size_t source_node: node.sources) {
        in_nodes.push_back(maps.node.at(source_node));
      }
      m_nodes.push_back(new ConcatenateNode(in_nodes));
      maps.node[iii] = m_nodes.back();
      return;
    }
    throw NNConfigurationException("unknown node type");
  }
}
namespace lwt {
  // graph
  Graph::Graph() {
    m_stacks.push_back(new Stack);
    Stack* stack = m_stacks.back();

    m_nodes.push_back(new InputNode(0, 2));
    INode* source1 = m_nodes.back();
    m_nodes.push_back(new InputNode(1, 2));
    INode* source2 = m_nodes.back();
    m_nodes.push_back(new ConcatenateNode({source1, source2}));
    INode* cat = m_nodes.back();
    m_nodes.push_back(new FeedForwardNode(stack, cat));
  }
  Graph::Graph(const std::vector<NodeConfig>& nodes,
               const std::vector<LayerConfig>& layers) {
    GraphMaps maps;
    for (size_t iii = 0; iii < nodes.size(); iii++) {
      build_node(iii, nodes, layers,
                 m_nodes, m_stacks, maps);
    }
    assert(maps.node.size() == nodes.size());
  }
  Graph::~Graph() {
    for (auto node: m_nodes) {
      delete node;
      node = 0;
    }
    for (auto stack: m_stacks) {
      delete stack;
      stack = 0;
    }
  }
  VectorXd Graph::compute(const ISource& source, size_t node_number) const {
    if (node_number >= m_nodes.size()) {
      throw NNEvaluationException(
        "Graph: no node at " + std::to_string(node_number));
    }
    return m_nodes.at(node_number)->compute(source);
  }
  VectorXd Graph::compute(const ISource& source) const {
    assert(m_nodes.size() > 0);
    return m_nodes.back()->compute(source);
  }
}
