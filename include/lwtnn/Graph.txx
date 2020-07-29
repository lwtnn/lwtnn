#ifndef GRAPH_TXX
#define GRAPH_TXX

#include "Graph.hh"
#include "lwtnn/Exceptions.hh"
#include "lwtnn/Stack.hh"

#include <set>
#include <memory>

namespace lwt {

  // Sources
  template<typename T>
  VectorSourceT<T>::VectorSourceT(std::vector<VectorX<T>>&& vv,
                                  std::vector<MatrixX<T>>&& mm):
    m_inputs(std::move(vv)),
    m_matrix_inputs(std::move(mm))
  {
  }
  
  template<typename T>
  VectorX<T> VectorSourceT<T>::at(size_t index) const {
    if (index >= m_inputs.size()) {
      throw NNEvaluationException(
        "VectorSource: no source vector defined at " + std::to_string(index));
    }
    return m_inputs.at(index);
  }
  
  template<typename T>
  MatrixX<T> VectorSourceT<T>::matrix_at(size_t index) const {
    if (index >= m_matrix_inputs.size()) {
      throw NNEvaluationException(
        "VectorSource: no source matrix defined at " + std::to_string(index));
    }
    return m_matrix_inputs.at(index);
  }
  
  template<typename T>
  DummySourceT<T>::DummySourceT(const std::vector<size_t>& input_sizes,
                                const std::vector<std::pair<size_t,size_t> >& ma):
    m_sizes(input_sizes),
    m_matrix_sizes(ma)
  {
  }
  
  template<typename T>
  VectorX<T> DummySourceT<T>::at(size_t index) const {
    if (index >= m_sizes.size()) {
      throw NNEvaluationException(
        "Dummy Source: no size defined at " + std::to_string(index));
    }
    size_t n_entries = m_sizes.at(index);
    VectorX<T> vec(n_entries);
    for (size_t iii = 0; iii < n_entries; iii++) {
      vec(iii) = iii;
    }
    return vec;
  }
  
  template<typename T>
  MatrixX<T> DummySourceT::matrix_at(size_t index) const {
    if (index >= m_matrix_sizes.size()) {
      throw NNEvaluationException(
        "Dummy Source: no size defined at " + std::to_string(index));
    }
    size_t n_rows = m_matrix_sizes.at(index).first;
    size_t n_cols = m_matrix_sizes.at(index).second;
    MatrixX<T> mat(n_rows, n_cols);
    for (size_t iii = 0; iii < n_rows; iii++) {
      for (size_t jjj = 0; jjj < n_cols; jjj++) {
        mat(iii, jjj) = jjj + n_cols * iii;
      }
    }
    return mat;
  }


  // Nodes
  template<typename T>
  InputNodeT<T>::InputNodeT(size_t index, size_t n_outputs):
    m_index(index),
    m_n_outputs(n_outputs)
  {
  }
  
  
  template<typename T>
  VectorX<T> InputNodeT::compute(const ISourceT<T>& source) const {
    VectorX<T> output = source.at(m_index);
    assert(output.rows() > 0);
    if (static_cast<size_t>(output.rows()) != m_n_outputs) {
      std::string len = std::to_string(output.rows());
      std::string found = std::to_string(m_n_outputs);
      throw NNEvaluationException(
        "Found vector of length " + len + ", expected " + found);
    }
    return output;
  }
  
  template<typename T>
  size_t InputNodeT<T>::n_outputs() const {
    return m_n_outputs;
  }

  template<typename T>
  FeedForwardNodeT<T>::FeedForwardNodeT(const StackT<T>* stack, const INodeT<T>* source):
    m_stack(stack),
    m_source(source)
  {
  }
  
  template<typename T>
  VectorX<T> FeedForwardNodeT<T>::compute(const ISourceT<T>& source) const {
    return m_stack->compute(m_source->compute(source));
  }
  
  template<typename T>
  size_t FeedForwardNodeT<T>::n_outputs() const {
    return m_stack->n_outputs();
  }

  template<typename T>
  ConcatenateNodeT<T>::ConcatenateNodeT(const std::vector<const INodeT<T>*>& sources):
    m_sources(sources),
    m_n_outputs(0)
  {
    for (const auto source: sources) {
      m_n_outputs += source->n_outputs();
    }
  }

  template<typename T>
  VectorX<T> ConcatenateNodeT<T>::compute(const ISourceT<T>& source) const {
    VectorX<T> output(m_n_outputs);
    size_t offset = 0;
    for (const auto node: m_sources) {
      VectorX<T> input = node->compute(source);
      size_t n_elements = input.rows();
      assert(n_elements == node->n_outputs());
      output.segment(offset, n_elements) = input;
      offset += n_elements;
    }
    assert(offset == m_n_outputs);
    return output;
  }
  
  template<typename T>
  size_t ConcatenateNodeT<T>::n_outputs() const {
    return m_n_outputs;
  }

  // Sequence nodes
  template<typename T>
  InputSequenceNodeT<T>::InputSequenceNodeT(size_t index, size_t n_outputs):
    m_index(index),
    m_n_outputs(n_outputs)
  {
  }
  
  template<typename T>
  MatrixX<T> InputSequenceNodeT<T>::scan(const ISourceT<T>& source) const {
    MatrixX<T> output = source.matrix_at(m_index);
    if (output.rows() == 0) {
      throw NNEvaluationException("empty input sequence");
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
  
  template<typename T>
  SequenceNode<T>::SequenceNodeT(const RecurrentStackT<T>* stack,
                                 const ISequenceNodeT<T>* source) :
    m_stack(stack),
    m_source(source)
  {
  }
  
  template<typename T>
  MatrixX<T> SequenceNodeT<T>::scan(const ISourceT<T>& source) const {
    return m_stack->scan(m_source->scan(source));
  }
  
  template<typename T>
  VectorX<T> SequenceNodeT<T>::compute(const ISourceT<T>& src) const {
    MatrixX<T> mat = scan(src);
    size_t n_cols = mat.cols();
    // special handling for empty sequence
    if (n_cols == 0) {
      return MatrixX<T>::Zero(mat.rows(), 1);
    }
    return mat.col(n_cols - 1);
  }
  
  template<typename T>
  size_t SequenceNodeT<T>::n_outputs() const {
    return m_stack->n_outputs();
  }

  template<typename T>
  TimeDistributedNodeT<T>::TimeDistributedNodeT(const StackT<T>* stack,
                                                const ISequenceNodeT<T>* source):
    m_stack(stack),
    m_source(source)
  {
  }
  
  template<typename T>
  MatrixX<T> TimeDistributedNodeT<T>::scan(const ISourceT<T>& source) const {
    MatrixXd input = m_source->scan(source);
    MatrixXd output(m_stack->n_outputs(), input.cols());
    size_t n_columns = input.cols();
    for (size_t col_n = 0; col_n < n_columns; col_n++) {
      output.col(col_n) = m_stack->compute(input.col(col_n));
    }
    return output;
  }
  
  template<typename T>
  size_t TimeDistributedNodeT<T>::n_outputs() const {
    return m_stack->n_outputs();
  }

  template<typename T>
  SumNodeT<T>::SumNodeT(const ISequenceNodeT<T>* source):
    m_source(source)
  {
  }
  
  template<typename T>
  VectorX<T> SumNodeT<T>::compute(const ISourceT<T>& source) const {
    return m_source->scan(source).rowwise().sum();
  }
  
  template<typename T>
  size_t SumNodeT<T>::n_outputs() const {
    return m_source->n_outputs();
  }
}

namespace {
  using namespace lwt;
  void throw_cfg(std::string msg, size_t index) {
    throw NNConfigurationException(msg + " " + std::to_string(index));
  }
  void check_compute_node(const NodeConfig& node) {
    size_t n_source = node.sources.size();
    if (n_source != 1) throw_cfg("need one source, found", n_source);
    int layer_n = node.index;
    if (layer_n < 0) throw_cfg("negative layer number", layer_n);
  }
  void check_compute_node(const NodeConfig& node, size_t n_layers) {
    check_compute_node(node);
    int layer_n = node.index;
    if (static_cast<size_t>(layer_n) >= n_layers) {
      throw_cfg("no layer number", layer_n);
    }
  }
  // NOTE: you own this pointer!
  INode* get_feedforward_node(
    const NodeConfig& node,
    const std::vector<LayerConfig>& layers,
    const std::unordered_map<size_t, INode*>& node_map,
    std::unordered_map<size_t, Stack*>& stack_map) {

    // FIXME: merge this block with the time distributed one later on
    check_compute_node(node, layers.size());
    INode* source = node_map.at(node.sources.at(0));
    int layer_n = node.index;
    if (!stack_map.count(layer_n)) {
      stack_map[layer_n] = new Stack(source->n_outputs(),
                                     {layers.at(layer_n)});
    }
    return new FeedForwardNode(stack_map.at(layer_n), source);
  }
  SequenceNode* get_sequence_node(
    const NodeConfig& node,
    const std::vector<LayerConfig>& layers,
    const std::unordered_map<size_t, ISequenceNode*>& node_map,
    std::unordered_map<size_t, RecurrentStack*>& stack_map) {

    check_compute_node(node, layers.size());
    ISequenceNode* source = node_map.at(node.sources.at(0));
    int layer_n = node.index;
    if (!stack_map.count(layer_n)) {
      stack_map[layer_n] = new RecurrentStack(source->n_outputs(),
                                              {layers.at(layer_n)});
    }
    return new SequenceNode(stack_map.at(layer_n), source);
  }
  TimeDistributedNode* get_time_distributed_node(
    const NodeConfig& node,
    const std::vector<LayerConfig>& layers,
    const std::unordered_map<size_t, ISequenceNode*>& node_map,
    std::unordered_map<size_t, Stack*>& stack_map) {

    // FIXME: merge this block with the FF block above
    check_compute_node(node, layers.size());
    ISequenceNode* source = node_map.at(node.sources.at(0));
    int layer_n = node.index;
    if (!stack_map.count(layer_n)) {
      stack_map[layer_n] = new Stack(source->n_outputs(),
                                     {layers.at(layer_n)});
    }
    return new TimeDistributedNode(stack_map.at(layer_n), source);
  }
}

namespace lwt {
  // graph
  Graph::Graph() {
    m_stacks[0] = new Stack;

    m_nodes[0] = new InputNode(0, 2);
    m_nodes[1] = new InputNode(1, 2);
    m_nodes[2] = new ConcatenateNode({m_nodes.at(0), m_nodes.at(1)});
    m_nodes[3] = new FeedForwardNode(m_stacks.at(0), m_nodes.at(2));
    m_last_node = 3;
  }
  Graph::Graph(const std::vector<NodeConfig>& nodes,
               const std::vector<LayerConfig>& layers):
    m_last_node(0)
  {
    for (size_t iii = 0; iii < nodes.size(); iii++) {
      build_node(iii, nodes, layers);
    }
    // assert(maps.node.size() + maps.seq_node.size() == nodes.size());
  }
  Graph::~Graph() {
    for (auto node: m_nodes) {
      delete node.second;
      node.second = nullptr;
    }
    for (auto node: m_seq_nodes) {
      // The m_nodes collection is the owner of anything that inherits
      // from both INode and ISequenceNode. So we try not to delete
      // anything that the m_nodes would already take care of.
      if (!m_nodes.count(node.first)) delete node.second;
      node.second = nullptr;
    }
    for (auto stack: m_stacks) {
      delete stack.second;
      stack.second = nullptr;
    }
    for (auto stack: m_seq_stacks) {
      delete stack.second;
      stack.second = nullptr;
    }
  }
  VectorXd Graph::compute(const ISource& source, size_t node_number) const {
    if (!m_nodes.count(node_number)) {
      auto num = std::to_string(node_number);
      if (m_seq_nodes.count(node_number)) {
        throw OutputRankException(
          "Graph: output at " + num + " not feed forward");
      }
      throw NNEvaluationException("Graph: no output at " + num);
    }
    return m_nodes.at(node_number)->compute(source);
  }
  VectorXd Graph::compute(const ISource& source) const {
    if (!m_nodes.count(m_last_node)) {
      throw OutputRankException("Graph: output is not a feed forward node");
    }
    return m_nodes.at(m_last_node)->compute(source);
  }
  MatrixXd Graph::scan(const ISource& source, size_t node_number) const {
    if (!m_seq_nodes.count(node_number)) {
      auto num = std::to_string(node_number);
      if (m_nodes.count(node_number)) {
        throw OutputRankException(
          "Graph: output at " + num + " not a sequence");
      }
      throw NNEvaluationException("Graph: no output at " + num);
    }
    return m_seq_nodes.at(node_number)->scan(source);
  }
  MatrixXd Graph::scan(const ISource& source) const {
    if (!m_seq_nodes.count(m_last_node)) {
      throw OutputRankException("Graph: output is not a sequence node");
    }
    return m_seq_nodes.at(m_last_node)->scan(source);
  }

  // ______________________________________________________________________
  // private methods

  void Graph::build_node(const size_t iii,
                         const std::vector<NodeConfig>& nodes,
                         const std::vector<LayerConfig>& layers,
                         std::set<size_t> cycle_check) {
    if (m_nodes.count(iii) || m_seq_nodes.count(iii)) return;

    // we insist that the upstream nodes are built before the
    // downstream ones, so the last node built should be some kind of
    // sink for graphs with only one output this will be it.
    m_last_node = iii;

    if (iii >= nodes.size()) throw_cfg("no node index", iii);

    const NodeConfig& node = nodes.at(iii);

    // if it's an input, build and return
    if (node.type == NodeConfig::Type::INPUT) {
      check_compute_node(node);
      size_t input_number = node.sources.at(0);
      m_nodes[iii] = new InputNode(input_number, node.index);
      return;
    } else if (node.type == NodeConfig::Type::INPUT_SEQUENCE) {
      check_compute_node(node);
      size_t input_number = node.sources.at(0);
      m_seq_nodes[iii] = new InputSequenceNode(input_number, node.index);
      return;
    }

    // otherwise build all the inputs first
    if (cycle_check.count(iii)) {
      throw NNConfigurationException("found cycle in graph");
    }
    cycle_check.insert(iii);
    for (size_t source_node: node.sources) {
      build_node(source_node, nodes, layers, cycle_check);
    }

    // check node types
    if (node.type == NodeConfig::Type::FEED_FORWARD) {
      m_nodes[iii] = get_feedforward_node(node, layers,
                                          m_nodes, m_stacks);
    } else if (node.type == NodeConfig::Type::TIME_DISTRIBUTED) {
      m_seq_nodes[iii] = get_time_distributed_node(node, layers,
                                                   m_seq_nodes, m_stacks);
    } else if (node.type == NodeConfig::Type::SEQUENCE) {
      std::unique_ptr<SequenceNode> seq_node(
        get_sequence_node(node, layers, m_seq_nodes, m_seq_stacks));
      // entering in m_nodes means that m_nodes will delete this one
      m_nodes[iii] = nullptr;
      m_seq_nodes[iii] = seq_node.get();
      m_nodes[iii] = seq_node.release();
    } else if (node.type == NodeConfig::Type::CONCATENATE) {
      // build concatenate layer
      std::vector<const INode*> in_nodes;
      for (size_t source_node: node.sources) {
        in_nodes.push_back(m_nodes.at(source_node));
      }
      m_nodes[iii] = new ConcatenateNode(in_nodes);
    } else if (node.type == NodeConfig::Type::SUM) {
      if (node.sources.size() != 1) {
        throw NNConfigurationException("Sum node needs exactly 1 source");
      }
      m_nodes[iii] = new SumNode(m_seq_nodes.at(node.sources.at(0)));
    } else {
      throw NNConfigurationException("unknown node type");
    }
  }

}

#endif
