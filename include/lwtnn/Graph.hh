#ifndef GRAPH_HH
#define GRAPH_HH

#include "NNLayerConfig.hh"
#include "Source.hh"

#include <Eigen/Dense>

#include <vector>
#include <unordered_map>
#include <set>

namespace lwt {
    
  template<typename T>
  using VectorX = Eigen::Matrix<T, Eigen::Dynamic, 1>;
  
  template<typename T>
  using MatrixX = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
  
  template<typename T>
  using ArrayX = Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic>;
  

  template<typename T>
  class StackT;
  
  using Stack = StackT<double>;
  
  template<typename T>
  class RecurrentStackT;
  
  using RecurrentStack = RecurrentStackT<double>;

  // node class: will return a VectorXd from ISource
  
  template<typename T>
  class INodeT
  {
  public:
    virtual ~INodeT() {}
    virtual VectorX<T> compute(const ISourceT<T>&) const = 0;
    virtual size_t n_outputs() const = 0;
  };
  
  using INode = INodeT<double>;

  template<typename T>
  class InputNodeT: public INodeT<T>
  {
  public:
    InputNodeT(size_t index, size_t n_outputs);
    virtual VectorX<T> compute(const ISourceT<T>&) const override;
    virtual size_t n_outputs() const override;
  private:
    size_t m_index;
    size_t m_n_outputs;
  };
  
  using InputNode = InputNodeT<double>;

  template<typename T>
  class FeedForwardNodeT: public INodeT<T>
  {
  public:
    FeedForwardNodeT(const StackT<T>*, const INodeT<T>* source);
    virtual VectorX<T> compute(const ISourceT<T>&) const override;
    virtual size_t n_outputs() const override;
  private:
    const StackT<T>* m_stack;
    const INodeT<T>* m_source;
  };
  
  using FeedForwardNode = FeedForwardNodeT<double>;

  template<typename T>
  class ConcatenateNodeT: public INodeT<T>
  {
  public:
    ConcatenateNodeT(const std::vector<const INodeT<T>*>&);
    virtual VectorX<T> compute(const ISourceT<T>&) const override;
    virtual size_t n_outputs() const override;
  private:
    std::vector<const INodeT<T>*> m_sources;
    size_t m_n_outputs;
  };
  
  using ConcatenateNode = ConcatenateNodeT<double>;

  // sequence nodes
  template<typename T>
  class ISequenceNodeT
  {
  public:
    virtual ~ISequenceNodeT() {}
    virtual MatrixX<T> scan(const ISource&) const = 0;
    virtual size_t n_outputs() const = 0;
  };
  
  using ISequenceNode = ISequenceNodeT<double>;

  template<typename T>
  class InputSequenceNodeT: public ISequenceNodeT<T>
  {
  public:
    InputSequenceNodeT(size_t index, size_t n_outputs);
    virtual MatrixX<T> scan(const ISourceT<T>&) const override;
    virtual size_t n_outputs() const override;
  private:
    size_t m_index;
    size_t m_n_outputs;
  };
  
  using InputSequenceNode = InputSequenceNodeT<double>;

  template<typename T>
  class SequenceNodeT: public ISequenceNodeT<T>, public INodeT<T>
  {
  public:
    SequenceNodeT(const RecurrentStackT<T>*, const ISequenceNodeT<T>* source);
    virtual MatrixX<T> scan(const ISource&) const override;
    virtual VectorX<T> compute(const ISource&) const override;
    virtual size_t n_outputs() const override;
  private:
    const RecurrentStackT<T>* m_stack;
    const ISequenceNodeT<T>* m_source;
  };
  
  using SequenceNode = SequenceNodeT<double>;
  
  template<typename T>
  class TimeDistributedNodeT: public ISequenceNodeT<T>
  {
  public:
    TimeDistributedNodeT(const StackT<T>*, const ISequenceNodeT<T>* source);
    virtual MatrixX<T> scan(const ISource&) const override;
    virtual size_t n_outputs() const override;
  private:
    const StackT<T>* m_stack;
    const ISequenceNodeT<T>* m_source;
  };
  
  using TimeDistributedNode = TimeDistributedNodeT<double>;
  
  template<typename T>
  class SumNodeT : public INodeT<T>
  {
  public:
    SumNodeT(const ISequenceNodeT<T>* source);
    virtual VectorX<T> compute(const ISourceT<T>&) const override;
    virtual size_t n_outputs() const override;
  private:
    const ISequenceNodeT<T>* m_source;
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
    VectorX<T> compute(const ISourceT<T>&, size_t node_number) const;
    VectorX<T> compute(const ISourceT<T>&) const;
    MatrixX<T> scan(const ISourceT<T>&, size_t node_number) const;
    MatrixX<T> scan(const ISourceT<T>&) const;
  private:
    void build_node(const size_t,
                    const std::vector<NodeConfig>& nodes,
                    const std::vector<LayerConfig>& layers,
                    std::set<size_t> cycle_check = {});

    std::unordered_map<size_t, INodeT<T>*> m_nodes;
    size_t m_last_node; // <-- convenience for graphs with one output
    std::unordered_map<size_t, StackT<T>*> m_stacks;
    std::unordered_map<size_t, ISequenceNodeT<T>*> m_seq_nodes;
    std::unordered_map<size_t, RecurrentStackT<T>*> m_seq_stacks;
    // At some point maybe also convolutional nodes, but we'd have to
    // have a use case for that first.
  };
  
  using Graph = GraphT<double>;
}

#include "Graph.txx"

#endif // GRAPH_HH
