#ifndef GRAPH_HH
#define GRAPH_HH

#include "NNLayerConfig.hh"
#include "Source.hh"

#include <Eigen/Dense>

#include <vector>
#include <unordered_map>
#include <set>

namespace lwt {

  class Stack;
  class RecurrentStack;


  // node class: will return a VectorXd from ISource
  class INode
  {
  public:
    virtual ~INode() {}
    virtual VectorXd compute(const ISource&) const = 0;
    virtual size_t n_outputs() const = 0;
  };

  class InputNode: public INode
  {
  public:
    InputNode(size_t index, size_t n_outputs);
    virtual VectorXd compute(const ISource&) const;
    virtual size_t n_outputs() const;
  private:
    size_t m_index;
    size_t m_n_outputs;
  };

  class FeedForwardNode: public INode
  {
  public:
    FeedForwardNode(const Stack*, const INode* source);
    virtual VectorXd compute(const ISource&) const;
    virtual size_t n_outputs() const;
  private:
    const Stack* m_stack;
    const INode* m_source;
  };

  class ConcatenateNode: public INode
  {
  public:
    ConcatenateNode(const std::vector<const INode*>&);
    virtual VectorXd compute(const ISource&) const;
    virtual size_t n_outputs() const;
  private:
    std::vector<const INode*> m_sources;
    size_t m_n_outputs;
  };

  // sequence nodes
  class ISequenceNode
  {
  public:
    virtual ~ISequenceNode() {}
    virtual MatrixXd scan(const ISource&) const = 0;
    virtual size_t n_outputs() const = 0;
  };

  class InputSequenceNode: public ISequenceNode
  {
  public:
    InputSequenceNode(size_t index, size_t n_outputs);
    virtual MatrixXd scan(const ISource&) const;
    virtual size_t n_outputs() const;
  private:
    size_t m_index;
    size_t m_n_outputs;
  };

  class SequenceNode: public ISequenceNode, public INode
  {
  public:
    SequenceNode(const RecurrentStack*, const ISequenceNode* source);
    virtual MatrixXd scan(const ISource&) const;
    virtual VectorXd compute(const ISource&) const;
    virtual size_t n_outputs() const;
  private:
    const RecurrentStack* m_stack;
    const ISequenceNode* m_source;
  };

  // Graph class, owns the nodes
  class Graph
  {
  public:
    Graph();                    // dummy constructor
    Graph(const std::vector<NodeConfig>& nodes,
          const std::vector<LayerConfig>& layers);
    Graph(Graph&) = delete;
    Graph& operator=(Graph&) = delete;
    ~Graph();
    VectorXd compute(const ISource&, size_t node_number) const;
    VectorXd compute(const ISource&) const;
  private:
    void build_node(const size_t,
                    const std::vector<NodeConfig>& nodes,
                    const std::vector<LayerConfig>& layers,
                    std::set<size_t> cycle_check = {});

    std::unordered_map<size_t, INode*> m_nodes;
    size_t m_last_node; // <-- convenience for graphs with one output
    std::unordered_map<size_t, Stack*> m_stacks;
    std::unordered_map<size_t, ISequenceNode*> m_seq_nodes;
    std::unordered_map<size_t, RecurrentStack*> m_seq_stacks;
    // At some point maybe also convolutional nodes, but we'd have to
    // have a use case for that first.
  };
}

#endif // GRAPH_HH
