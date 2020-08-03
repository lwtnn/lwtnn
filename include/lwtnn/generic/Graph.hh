#ifndef GENERIC_GRAPH_HH
#define GENERIC_GRAPH_HH

#include <vector>
#include <unordered_map>
#include <set>

#include "lwtnn/NNLayerConfig.hh"
#include "lwtnn/Source.hh"
#include "eigen_typedefs.hh"

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
    virtual size_t n_outputs() const = 0;
  };

  template<typename T>
  class InputNode: public INode<T>
  {
  public:
    InputNode(size_t index, size_t n_outputs);
    virtual VectorX<T> compute(const ISource<T>&) const override;
    virtual size_t n_outputs() const override;
  private:
    size_t m_index;
    size_t m_n_outputs;
  };

  template<typename T>
  class FeedForwardNode: public INode<T>
  {
  public:
    FeedForwardNode(const Stack<T>*, const INode<T>* source);
    virtual VectorX<T> compute(const ISource<T>&) const override;
    virtual size_t n_outputs() const override;
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
    virtual size_t n_outputs() const override;
  private:
    std::vector<const INode<T>*> m_sources;
    size_t m_n_outputs;
  };

  // sequence nodes
  template<typename T>
  class ISequenceNode
  {
  public:
    virtual ~ISequenceNode() {}
    virtual MatrixX<T> scan(const ISource&) const = 0;
    virtual size_t n_outputs() const = 0;
  };

  template<typename T>
  class InputSequenceNode: public ISequenceNode<T>
  {
  public:
    InputSequenceNode(size_t index, size_t n_outputs);
    virtual MatrixX<T> scan(const ISource<T>&) const override;
    virtual size_t n_outputs() const override;
  private:
    size_t m_index;
    size_t m_n_outputs;
  };

  template<typename T>
  class SequenceNode: public ISequenceNode<T>, public INode<T>
  {
  public:
    SequenceNode(const RecurrentStack<T>*, const ISequenceNode<T>* source);
    virtual MatrixX<T> scan(const ISource<T>&) const override;
    virtual VectorX<T> compute(const ISource<T>&) const override;
    virtual size_t n_outputs() const override;
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
    virtual size_t n_outputs() const override;
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
    virtual size_t n_outputs() const override;
  private:
    const ISequenceNode<T>* m_source;
  };

  // Graph class, owns the nodes
  template<typename T>
  class GraphT
  {
  public:
    GraphT();                    // dummy constructor
    GraphT(const std::vector<NodeConfig>& nodes,
          const std::vector<LayerConfig>& layers);
    GraphT(GraphT&) = delete;
    GraphT& operator=(GraphT&) = delete;
    ~GraphT();
    VectorX<T> compute(const ISource<T>&, size_t node_number) const;
    VectorX<T> compute(const ISource<T>&) const;
    MatrixX<T> scan(const ISource<T>&, size_t node_number) const;
    MatrixX<T> scan(const ISource<T>&) const;
  private:
    void build_node(const size_t,
                    const std::vector<NodeConfig>& nodes,
                    const std::vector<LayerConfig>& layers,
                    std::set<size_t> cycle_check = {});

    std::unordered_map<size_t, INode<T>*> m_nodes;
    size_t m_last_node; // <-- convenience for graphs with one output
    std::unordered_map<size_t, Stack<T>*> m_stacks;
    std::unordered_map<size_t, ISequenceNode<T>*> m_seq_nodes;
    std::unordered_map<size_t, RecurrentStack<T>*> m_seq_stacks;
    // At some point maybe also convolutional nodes, but we'd have to
    // have a use case for that first.
  };
  
} // namespace generic
  
  using INode = generic::INode<double>;
  using FeedForwardNode = generic::FeedForwardNode<double>;
  using InputNode = generic::InputNode<double>;
  using ConcatenateNode = generic::ConcatenateNode<double>;
  using ISequenceNode = generic::ISequenceNode<double>;
  using InputSequenceNode = generic::InputSequenceNode<double>;
  using SequenceNode = generic::SequenceNode<double>;
  using TimeDistributedNode = generic::TimeDistributedNode<double>;
  
  using Graph = Graph<double>;
  
} // namespace lwt

#include "Graph.txx"

#endif // GRAPH_HH
