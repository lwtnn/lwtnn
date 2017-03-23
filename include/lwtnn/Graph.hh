#ifndef GRAPH_HH
#define GRAPH_HH

#include <Eigen/Dense>

#include <vector>

namespace lwt {

  class Stack;

  using Eigen::VectorXd;
  using Eigen::MatrixXd;

  // this is called by input nodes to get the inputs
  class ISource
  {
  public:
    virtual VectorXd at(size_t index) const = 0;
  };

  class VectorSource: public ISource
  {
  public:
    VectorSource(const std::vector<VectorXd>&&);
    virtual VectorXd at(size_t index) const;
  private:
    std::vector<VectorXd> m_inputs;
  };

  class DummySource: public ISource
  {
  public:
    DummySource(const std::vector<size_t>& input_sizes);
    virtual VectorXd at(size_t index) const;
  private:
    std::vector<size_t> m_sizes;
  };


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


  // Graph class, owns the nodes
  class Graph
  {
  public:
    Graph();                    // dummy constructor
    Graph(Graph&) = delete;
    Graph& operator=(Graph&) = delete;
    ~Graph();
    VectorXd compute(const ISource&, size_t node_number) const;
    VectorXd compute(const ISource&) const;
  private:
    std::vector<INode*> m_nodes;
    std::vector<Stack*> m_stacks;
  };
}

#endif // GRAPH_HH
