#ifndef LWTNN_GENERIC_GRAPH_HH
#define LWTNN_GENERIC_GRAPH_HH

#include "lwtnn/generic/eigen_typedefs.hh"
#include "lwtnn/generic/Source.hh"
#include "lwtnn/NNLayerConfig.hh"

#include <vector>
#include <unordered_map>
#include <set>

namespace lwt {
namespace generic {

  // Forward declaretions
  template<typename T>
  class Stack;

  template<typename T>
  class RecurrentStack;

  // Node class: will return a VectorXd from ISource
  template<typename T>
  class INode
  {
  public:
    virtual ~INode() {}
    virtual VectorX<T> compute(const ISource<T>&) const = 0;
    virtual std::size_t n_outputs() const = 0;
  };

  template<typename T>
  class InputNode: public INode<T>
  {
  public:
    InputNode(std::size_t index, std::size_t n_outputs);
    virtual VectorX<T> compute(const ISource<T>&) const override;
    virtual std::size_t n_outputs() const override;
  private:
    std::size_t m_index;
    std::size_t m_n_outputs;
  };

  template<typename T>
  class FeedForwardNode: public INode<T>
  {
  public:
    FeedForwardNode(const Stack<T>*, const INode<T>* source);
    virtual VectorX<T> compute(const ISource<T>&) const override;
    virtual std::size_t n_outputs() const override;
  private:
    const Stack<T>* m_stack;
    const INode<T>* m_source;
  };

  template<typename T>
  class ConcatenateNode: public INode<T>
  {
  public:
    ConcatenateNode(const std::vector<const INode<T>*>&);
    virtual VectorX<T> compute(const ISource<T>&) const override;
    virtual std::size_t n_outputs() const override;
  private:
    std::vector<const INode<T>*> m_sources;
    std::size_t m_n_outputs;
  };

  template<typename T>
  class AddNode: public INode<T>
  {
  public:
    AddNode(const std::vector<const INode<T>*>&);
    virtual VectorX<T> compute(const ISource<T>&) const override;
    virtual std::size_t n_outputs() const override;
  private:
    std::vector<const INode<T>*> m_sources;
    std::size_t m_n_outputs;
  };

  // sequence nodes
  template<typename T>
  class ISequenceNode
  {
  public:
    virtual ~ISequenceNode() {}
    virtual MatrixX<T> scan(const ISource<T>&) const = 0;
    virtual std::size_t n_outputs() const = 0;
  };

  template<typename T>
  class InputSequenceNode: public ISequenceNode<T>
  {
  public:
    InputSequenceNode(std::size_t index, std::size_t n_outputs);
    virtual MatrixX<T> scan(const ISource<T>&) const override;
    virtual std::size_t n_outputs() const override;
  private:
    std::size_t m_index;
    std::size_t m_n_outputs;
  };

  template<typename T>
  class SequenceNode: public ISequenceNode<T>, public INode<T>
  {
  public:
    SequenceNode(const RecurrentStack<T>*, const ISequenceNode<T>* source);
    virtual MatrixX<T> scan(const ISource<T>&) const override;
    virtual VectorX<T> compute(const ISource<T>&) const override;
    virtual std::size_t n_outputs() const override;
  private:
    const RecurrentStack<T>* m_stack;
    const ISequenceNode<T>* m_source;
  };

  template<typename T>
  class TimeDistributedNode: public ISequenceNode<T>
  {
  public:
    TimeDistributedNode(const Stack<T>*, const ISequenceNode<T>* source);
    virtual MatrixX<T> scan(const ISource<T>&) const override;
    virtual std::size_t n_outputs() const override;
  private:
    const Stack<T>* m_stack;
    const ISequenceNode<T>* m_source;
  };

  template<typename T>
  class SumNode: public INode<T>
  {
  public:
    SumNode(const ISequenceNode<T>* source);
    virtual VectorX<T> compute(const ISource<T>&) const override;
    virtual std::size_t n_outputs() const override;
  private:
    const ISequenceNode<T>* m_source;
  };

  // Graph class, owns the nodes
  template<typename T>
  class Graph
  {
  public:
    Graph();                    // dummy constructor
    Graph(const std::vector<NodeConfig>& nodes,
          const std::vector<LayerConfig>& layers);
    Graph(Graph&) = delete;
    Graph& operator=(Graph&) = delete;
    ~Graph();
    VectorX<T> compute(const ISource<T>&, std::size_t node_number) const;
    VectorX<T> compute(const ISource<T>&) const;
    MatrixX<T> scan(const ISource<T>&, std::size_t node_number) const;
    MatrixX<T> scan(const ISource<T>&) const;
  private:
    void build_node(const std::size_t,
                    const std::vector<NodeConfig>& nodes,
                    const std::vector<LayerConfig>& layers,
                    std::set<std::size_t> cycle_check = {});

    std::unordered_map<std::size_t, INode<T>*> m_nodes;
    std::size_t m_last_node; // <-- convenience for graphs with one output
    std::unordered_map<std::size_t, Stack<T>*> m_stacks;
    std::unordered_map<std::size_t, ISequenceNode<T>*> m_seq_nodes;
    std::unordered_map<std::size_t, RecurrentStack<T>*> m_seq_stacks;
    // At some point maybe also convolutional nodes, but we'd have to
    // have a use case for that first.
  };

} // namespace generic

} // namespace lwt

#include "Graph.tcc"

#endif // LWTNN_GENERIC_GRAPH_HH
