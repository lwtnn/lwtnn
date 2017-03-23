#include "lwtnn/Graph.hh"
#include "lwtnn/Exceptions.hh"
#include "lwtnn/Stack.hh"

namespace lwt {

  // Sources
  VectorSource::VectorSource(const std::vector<VectorXd>&& vv):
    m_inputs(std::move(vv))
  {
  }
  VectorXd VectorSource::at(size_t index) const {
    if (index >= m_inputs.size()) {
      throw NNEvaluationException(
        "VectorSource: no source vector defined at " + std::to_string(index));
    }
    return m_inputs.at(index);
  };

  DummySource::DummySource(const std::vector<size_t>& input_sizes):
    m_sizes(input_sizes)
  {
  }
  VectorXd DummySource::at(size_t index) const {
    if (index >= m_sizes.size()) {
      throw NNEvaluationException(
        "Dummy Source: no size defined at " + std::to_string(index));
    }
    size_t n_entries = m_sizes.at(index);
    VectorXd vec(n_entries);
    for (int iii = 0; iii < n_entries; iii++) {
      vec(iii) = iii;
    }
    return vec;
  }


  // Nodes
  InputNode::InputNode(size_t index):
    m_index(index)
  {
  }
  VectorXd InputNode::compute(const ISource& source) const {
    return source.at(m_index);
  }

  FeedForwardNode::FeedForwardNode(const Stack* stack, const INode* source):
    m_stack(stack),
    m_source(source)
  {
  }
  VectorXd FeedForwardNode::compute(const ISource& source) const {
    return m_stack->compute(m_source->compute(source));
  }

  // graph
  Graph::Graph() {
    m_stacks.push_back(new Stack);
    Stack* stack = m_stacks.back();

    m_nodes.push_back(new InputNode(0));
    INode* source = m_nodes.back();
    m_nodes.push_back(new FeedForwardNode(stack, source));
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
}
