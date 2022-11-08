#ifndef LWTNN_GENERIC_GRAPH_TCC
#define LWTNN_GENERIC_GRAPH_TCC

#include "lwtnn/generic/Stack.hh"
#include "lwtnn/generic/Graph.hh"
#include "lwtnn/Exceptions.hh"

#include <set>
#include <memory>

namespace lwt {
namespace generic {

  // Sources
  template<typename T>
  VectorSource<T>::VectorSource(std::vector<VectorX<T>>&& vv,
                                  std::vector<MatrixX<T>>&& mm):
    m_inputs(std::move(vv)),
    m_matrix_inputs(std::move(mm))
  {
  }

  template<typename T>
  VectorX<T> VectorSource<T>::at(std::size_t index) const {
    if (index >= m_inputs.size()) {
      throw NNEvaluationException(
        "VectorSource: no source vector defined at " + std::to_string(index));
    }
    return m_inputs.at(index);
  }

  template<typename T>
  MatrixX<T> VectorSource<T>::matrix_at(std::size_t index) const {
    if (index >= m_matrix_inputs.size()) {
      throw NNEvaluationException(
        "VectorSource: no source matrix defined at " + std::to_string(index));
    }
    return m_matrix_inputs.at(index);
  }

  template<typename T>
  DummySource<T>::DummySource(const std::vector<std::size_t>& input_sizes,
                                const std::vector<std::pair<std::size_t,std::size_t> >& ma):
    m_sizes(input_sizes),
    m_matrix_sizes(ma)
  {
  }

  template<typename T>
  VectorX<T> DummySource<T>::at(std::size_t index) const {
    if (index >= m_sizes.size()) {
      throw NNEvaluationException(
        "Dummy Source: no size defined at " + std::to_string(index));
    }
    std::size_t n_entries = m_sizes.at(index);
    VectorX<T> vec(n_entries);
    for (std::size_t iii = 0; iii < n_entries; iii++) {
      vec(iii) = iii;
    }
    return vec;
  }

  template<typename T>
  MatrixX<T> DummySource<T>::matrix_at(std::size_t index) const {
    if (index >= m_matrix_sizes.size()) {
      throw NNEvaluationException(
        "Dummy Source: no size defined at " + std::to_string(index));
    }
    std::size_t n_rows = m_matrix_sizes.at(index).first;
    std::size_t n_cols = m_matrix_sizes.at(index).second;
    MatrixX<T> mat(n_rows, n_cols);
    for (std::size_t iii = 0; iii < n_rows; iii++) {
      for (std::size_t jjj = 0; jjj < n_cols; jjj++) {
        mat(iii, jjj) = jjj + n_cols * iii;
      }
    }
    return mat;
  }


  // Nodes
  template<typename T>
  InputNode<T>::InputNode(std::size_t index, std::size_t n_outputs):
    m_index(index),
    m_n_outputs(n_outputs)
  {
  }


  template<typename T>
  VectorX<T> InputNode<T>::compute(const ISource<T>& source) const {
    VectorX<T> output = source.at(m_index);
    assert(output.rows() > 0);
    if (static_cast<std::size_t>(output.rows()) != m_n_outputs) {
      std::string len = std::to_string(output.rows());
      std::string found = std::to_string(m_n_outputs);
      throw NNEvaluationException(
        "Found vector of length " + len + ", expected " + found);
    }
    return output;
  }

  template<typename T>
  std::size_t InputNode<T>::n_outputs() const {
    return m_n_outputs;
  }

  template<typename T>
  FeedForwardNode<T>::FeedForwardNode(const Stack<T>* stack, const INode<T>* source):
    m_stack(stack),
    m_source(source)
  {
  }

  template<typename T>
  VectorX<T> FeedForwardNode<T>::compute(const ISource<T>& source) const {
    return m_stack->compute(m_source->compute(source));
  }

  template<typename T>
  std::size_t FeedForwardNode<T>::n_outputs() const {
    return m_stack->n_outputs();
  }

  template<typename T>
  ConcatenateNode<T>::ConcatenateNode(const std::vector<const INode<T>*>& sources):
    m_sources(sources),
    m_n_outputs(0)
  {
    for (const auto source: sources) {
      m_n_outputs += source->n_outputs();
    }
  }

  template<typename T>
  VectorX<T> ConcatenateNode<T>::compute(const ISource<T>& source) const {
    VectorX<T> output(m_n_outputs);
    std::size_t offset = 0;
    for (const auto node: m_sources) {
      VectorX<T> input = node->compute(source);
      std::size_t n_elements = input.rows();
      assert(n_elements == node->n_outputs());
      output.segment(offset, n_elements) = input;
      offset += n_elements;
    }
    assert(offset == m_n_outputs);
    return output;
  }

  template<typename T>
  std::size_t ConcatenateNode<T>::n_outputs() const {
    return m_n_outputs;
  }

  template<typename T>
  AddNode<T>::AddNode(const std::vector<const INode<T>*>& sources):
    m_sources(sources)
  {
    if (sources.size() < 1){
      throw NNConfigurationException("Add layer must have sources");
    }
    m_n_outputs = sources[0]->n_outputs();
    //Check to make sure each input layer has the same size
    for (const auto source: sources) {
      if(source->n_outputs() != m_n_outputs){
        throw NNConfigurationException("All sources of an add layer must have same dimension");
      }
    }
  }

  template<typename T>
  VectorX<T> AddNode<T>::compute(const ISource<T>& source) const {
    VectorX<T> output = VectorX<T>::Zero(m_n_outputs);
    for (const auto node: m_sources) {
      VectorX<T> input = node->compute(source);
      assert((size_t)input.rows() == node->n_outputs());
      output += input;
    }
    return output;
  }

  template<typename T>
  std::size_t AddNode<T>::n_outputs() const {
    return m_n_outputs;
  }

  // Sequence nodes
  template<typename T>
  InputSequenceNode<T>::InputSequenceNode(std::size_t index, std::size_t n_outputs):
    m_index(index),
    m_n_outputs(n_outputs)
  {
  }

  template<typename T>
  MatrixX<T> InputSequenceNode<T>::scan(const ISource<T>& source) const {
    MatrixX<T> output = source.matrix_at(m_index);
    if (output.rows() == 0) {
      throw NNEvaluationException("empty input sequence");
    }
    if (static_cast<std::size_t>(output.rows()) != m_n_outputs) {
      std::string len = std::to_string(output.rows());
      std::string found = std::to_string(m_n_outputs);
      throw NNEvaluationException(
        "Found vector of length " + len + ", expected " + found);
    }
    return output;
  }
  template<typename T>
  std::size_t InputSequenceNode<T>::n_outputs() const {
    return m_n_outputs;
  }

  template<typename T>
  SequenceNode<T>::SequenceNode(const RecurrentStack<T>* stack,
                                 const ISequenceNode<T>* source) :
    m_stack(stack),
    m_source(source)
  {
  }

  template<typename T>
  MatrixX<T> SequenceNode<T>::scan(const ISource<T>& source) const {
    return m_stack->scan(m_source->scan(source));
  }

  template<typename T>
  VectorX<T> SequenceNode<T>::compute(const ISource<T>& src) const {
    MatrixX<T> mat = scan(src);
    std::size_t n_cols = mat.cols();
    // special handling for empty sequence
    if (n_cols == 0) {
      return MatrixX<T>::Zero(mat.rows(), 1);
    }
    return mat.col(n_cols - 1);
  }

  template<typename T>
  std::size_t SequenceNode<T>::n_outputs() const {
    return m_stack->n_outputs();
  }

  template<typename T>
  TimeDistributedNode<T>::TimeDistributedNode(const Stack<T>* stack,
                                                const ISequenceNode<T>* source):
    m_stack(stack),
    m_source(source)
  {
  }

  template<typename T>
  MatrixX<T> TimeDistributedNode<T>::scan(const ISource<T>& source) const {
    MatrixX<T> input = m_source->scan(source);
    MatrixX<T> output(m_stack->n_outputs(), input.cols());
    std::size_t n_columns = input.cols();
    for (std::size_t col_n = 0; col_n < n_columns; col_n++) {
      output.col(col_n) = m_stack->compute(input.col(col_n));
    }
    return output;
  }

  template<typename T>
  std::size_t TimeDistributedNode<T>::n_outputs() const {
    return m_stack->n_outputs();
  }

  template<typename T>
  SumNode<T>::SumNode(const ISequenceNode<T>* source):
    m_source(source)
  {
  }

  template<typename T>
  VectorX<T> SumNode<T>::compute(const ISource<T>& source) const {
    return m_source->scan(source).rowwise().sum();
  }

  template<typename T>
  std::size_t SumNode<T>::n_outputs() const {
    return m_source->n_outputs();
  }

  namespace {
    void throw_cfg(std::string msg, std::size_t index) {
        throw NNConfigurationException(msg + " " + std::to_string(index));
    }
    void check_compute_node(const NodeConfig& node) {
        std::size_t n_source = node.sources.size();
        if (n_source != 1) throw_cfg("need one source, found", n_source);
        int layer_n = node.index;
        if (layer_n < 0) throw_cfg("negative layer number", layer_n);
    }
    void check_compute_node(const NodeConfig& node, std::size_t n_layers) {
        check_compute_node(node);
        int layer_n = node.index;
        if (static_cast<std::size_t>(layer_n) >= n_layers) {
        throw_cfg("no layer number", layer_n);
        }
    }
  }

  // NOTE: you own this pointer!
  template<typename T>
  INode<T>* get_feedforward_node(
    const NodeConfig& node,
    const std::vector<LayerConfig>& layers,
    const std::unordered_map<std::size_t, INode<T>*>& node_map,
    std::unordered_map<std::size_t, Stack<T>*>& stack_map) {

    // FIXME: merge this block with the time distributed one later on
    check_compute_node(node, layers.size());
    INode<T>* source = node_map.at(node.sources.at(0));
    int layer_n = node.index;
    if (!stack_map.count(layer_n)) {
      stack_map[layer_n] = new Stack<T>(source->n_outputs(),
                                         {layers.at(layer_n)});
    }
    return new FeedForwardNode<T>(stack_map.at(layer_n), source);
  }
  template<typename T>
  SequenceNode<T>* get_sequence_node(
    const NodeConfig& node,
    const std::vector<LayerConfig>& layers,
    const std::unordered_map<std::size_t, ISequenceNode<T>*>& node_map,
    std::unordered_map<std::size_t, RecurrentStack<T>*>& stack_map) {

    check_compute_node(node, layers.size());
    ISequenceNode<T>* source = node_map.at(node.sources.at(0));
    int layer_n = node.index;
    if (!stack_map.count(layer_n)) {
      stack_map[layer_n] = new RecurrentStack<T>(source->n_outputs(),
                                              {layers.at(layer_n)});
    }
    return new SequenceNode<T>(stack_map.at(layer_n), source);
  }
  template<typename T>
  TimeDistributedNode<T>* get_time_distributed_node(
    const NodeConfig& node,
    const std::vector<LayerConfig>& layers,
    const std::unordered_map<std::size_t, ISequenceNode<T>*>& node_map,
    std::unordered_map<std::size_t, Stack<T>*>& stack_map) {

    // FIXME: merge this block with the FF block above
    check_compute_node(node, layers.size());
    ISequenceNode<T>* source = node_map.at(node.sources.at(0));
    int layer_n = node.index;
    if (!stack_map.count(layer_n)) {
      stack_map[layer_n] = new Stack<T>(source->n_outputs(),
                                     {layers.at(layer_n)});
    }
    return new TimeDistributedNode<T>(stack_map.at(layer_n), source);
  }

  // graph
  template<typename T>
  Graph<T>::Graph() {
    m_stacks[0] = new Stack<T>;

    m_nodes[0] = new InputNode<T>(0, 2);
    m_nodes[1] = new InputNode<T>(1, 2);
    m_nodes[2] = new ConcatenateNode<T>({m_nodes.at(0), m_nodes.at(1)});
    m_nodes[3] = new FeedForwardNode<T>(m_stacks.at(0), m_nodes.at(2));
    m_last_node = 3;
  }

  template<typename T>
  Graph<T>::Graph(const std::vector<NodeConfig>& nodes,
                    const std::vector<LayerConfig>& layers):
    m_last_node(0)
  {
    for (std::size_t iii = 0; iii < nodes.size(); iii++) {
      build_node(iii, nodes, layers);
    }
    // assert(maps.node.size() + maps.seq_node.size() == nodes.size());
  }
  template<typename T>
  Graph<T>::~Graph() {
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
  template<typename T>
  VectorX<T> Graph<T>::compute(const ISource<T>& source, std::size_t node_number) const {
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
  template<typename T>
  VectorX<T> Graph<T>::compute(const ISource<T>& source) const {
    if (!m_nodes.count(m_last_node)) {
      throw OutputRankException("Graph: output is not a feed forward node");
    }
    return m_nodes.at(m_last_node)->compute(source);
  }
  template<typename T>
  MatrixX<T> Graph<T>::scan(const ISource<T>& source, std::size_t node_number) const {
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
  template<typename T>
  MatrixX<T> Graph<T>::scan(const ISource<T>& source) const {
    if (!m_seq_nodes.count(m_last_node)) {
      throw OutputRankException("Graph: output is not a sequence node");
    }
    return m_seq_nodes.at(m_last_node)->scan(source);
  }

  // ______________________________________________________________________
  // private methods

  template<typename T>
  void Graph<T>::build_node(const std::size_t iii,
                         const std::vector<NodeConfig>& nodes,
                         const std::vector<LayerConfig>& layers,
                         std::set<std::size_t> cycle_check) {
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
      std::size_t input_number = node.sources.at(0);
      m_nodes[iii] = new InputNode<T>(input_number, node.index);
      return;
    } else if (node.type == NodeConfig::Type::INPUT_SEQUENCE) {
      check_compute_node(node);
      std::size_t input_number = node.sources.at(0);
      m_seq_nodes[iii] = new InputSequenceNode<T>(input_number, node.index);
      return;
    }

    // otherwise build all the inputs first
    if (cycle_check.count(iii)) {
      throw NNConfigurationException("found cycle in graph");
    }
    cycle_check.insert(iii);
    for (std::size_t source_node: node.sources) {
      build_node(source_node, nodes, layers, cycle_check);
    }

    // check node types
    if (node.type == NodeConfig::Type::FEED_FORWARD) {
      m_nodes[iii] = get_feedforward_node(node, layers,
                                          m_nodes, m_stacks);
    } else if (node.type == NodeConfig::Type::TIME_DISTRIBUTED) {
      m_seq_nodes[iii] = get_time_distributed_node<T>(node, layers,
                                                   m_seq_nodes, m_stacks);
    } else if (node.type == NodeConfig::Type::SEQUENCE) {
      std::unique_ptr<SequenceNode<T>> seq_node(
        get_sequence_node(node, layers, m_seq_nodes, m_seq_stacks));
      // entering in m_nodes means that m_nodes will delete this one
      m_nodes[iii] = nullptr;
      m_seq_nodes[iii] = seq_node.get();
      m_nodes[iii] = seq_node.release();
    } else if (node.type == NodeConfig::Type::CONCATENATE) {
      // build concatenate layer
      std::vector<const INode<T>*> in_nodes;
      for (std::size_t source_node: node.sources) {
        in_nodes.push_back(m_nodes.at(source_node));
      }
      m_nodes[iii] = new ConcatenateNode<T>(in_nodes);
    } else if (node.type == NodeConfig::Type::SUM) {
      if (node.sources.size() != 1) {
        throw NNConfigurationException("Sum node needs exactly 1 source");
      }
      m_nodes[iii] = new SumNode<T>(m_seq_nodes.at(node.sources.at(0)));
    } else if (node.type == NodeConfig::Type::ADD) {
      // build add layer
      std::vector<const INode<T>*> in_nodes;
      for (std::size_t source_node: node.sources) {
        in_nodes.push_back(m_nodes.at(source_node));
      }
      m_nodes[iii] = new AddNode<T>(in_nodes);
    }
    else {
      throw NNConfigurationException("unknown node type");
    }
  }

} // namespace generic
} // namespace lwt

#endif
