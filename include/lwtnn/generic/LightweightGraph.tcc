#ifndef LWTNN_GENERIC_LIGHTWEIGHT_GRAPH_TCC
#define LWTNN_GENERIC_LIGHTWEIGHT_GRAPH_TCC

#include "lwtnn/generic/LightweightGraph.hh"
#include "lwtnn/generic/InputPreprocessor.hh"
#include "lwtnn/generic/Graph.hh"
#include "lwtnn/generic/eigen_typedefs.hh"

namespace {
  using namespace lwt;

  // this is used internally to ensure that we only look up map inputs
  // when the network asks for them.
  template<typename T>
  class LazySource: public generic::ISource<T>
  {
    typedef typename generic::LightweightGraph<T>::NodeMap NodeMap;
    using IP = generic::InputPreprocessor<T>;
    typedef std::vector<std::pair<std::string, IP*> > Preprocs;
    typedef typename generic::LightweightGraph<T>::SeqNodeMap SeqNodeMap;
    using IVP = generic::InputVectorPreprocessor<T>;
    typedef std::vector<std::pair<std::string, IVP*> > VecPreprocs;
  public:
    LazySource(const NodeMap&, const SeqNodeMap&,
               const Preprocs&, const VecPreprocs&);
    virtual VectorX<T> at(std::size_t index) const override;
    virtual MatrixX<T> matrix_at(std::size_t index) const override;
  private:
    const NodeMap& m_nodes;
    const SeqNodeMap& m_seqs;
    const Preprocs& m_preprocs;
    const VecPreprocs& m_vec_preprocs;
  };

  template<typename T>
  LazySource<T>::LazySource(const NodeMap& n, const SeqNodeMap& s,
                            const Preprocs& p, const VecPreprocs& v):
    m_nodes(n), m_seqs(s), m_preprocs(p), m_vec_preprocs(v)
  {
  }

  template<typename T>
  VectorX<T> LazySource<T>::at(std::size_t index) const
  {
    const auto& proc = m_preprocs.at(index);
    if (!m_nodes.count(proc.first)) {
      throw NNEvaluationException("Can't find node " + proc.first);
    }
    const auto& preproc = *proc.second;
    return preproc(m_nodes.at(proc.first));
  }

  template<typename T>
  MatrixX<T> LazySource<T>::matrix_at(std::size_t index) const
  {
    const auto& proc = m_vec_preprocs.at(index);
    if (!m_seqs.count(proc.first)) {
      throw NNEvaluationException("Can't find sequence node " + proc.first);
    }
    const auto& preproc = *proc.second;
    return preproc(m_seqs.at(proc.first));
  }
}

namespace lwt {
namespace generic {
  // ______________________________________________________________________
  // Lightweight Graph

//   typedef LightweightGraph::NodeMap NodeMap;
  template<typename T>
  LightweightGraph<T>::LightweightGraph(const GraphConfig& config,
                                        std::string default_output):
    m_graph(new Graph<T>(config.nodes, config.layers))
  {
    for (const auto& node: config.inputs) {
      m_preprocs.emplace_back(
        node.name, new InputPreprocessor<T>(node.variables));
    }
    for (const auto& node: config.input_sequences) {
      m_vec_preprocs.emplace_back(
        node.name, new InputVectorPreprocessor<T>(node.variables));
    }
    std::size_t output_n = 0;
    for (const auto& node: config.outputs) {
      m_outputs.emplace_back(node.second.node_index, node.second.labels);
      m_output_indices.emplace(node.first, output_n);
      output_n++;
    }
    if (default_output.size() > 0) {
      if (!m_output_indices.count(default_output)) {
        throw NNConfigurationException("no output node" + default_output);
      }
      m_default_output = m_output_indices.at(default_output);
    } else if (output_n == 1) {
      m_default_output = 0;
    } else {
      throw NNConfigurationException("you must specify a default output");
    }
  }

  template<typename T>
  LightweightGraph<T>::~LightweightGraph() {
    delete m_graph;
    for (auto& preproc: m_preprocs) {
      delete preproc.second;
      preproc.second = 0;
    }
    for (auto& preproc: m_vec_preprocs) {
      delete preproc.second;
      preproc.second = 0;
    }
  }

  template<typename T>
  ValueMap LightweightGraph<T>::compute(const NodeMap& nodes,
                                     const SeqNodeMap& seq) const {
    return compute(nodes, seq, m_default_output);
  }

  template<typename T>
  ValueMap LightweightGraph<T>::compute(const NodeMap& nodes,
                                     const SeqNodeMap& seq,
                                     const std::string& output) const {
    if (!m_output_indices.count(output)) {
      throw NNEvaluationException("no output node " + output);
    }
    return compute(nodes, seq, m_output_indices.at(output));
  }

  template<typename T>
  ValueMap LightweightGraph<T>::compute(const NodeMap& nodes,
                                     const SeqNodeMap& seq,
                                     std::size_t idx) const {
    LazySource<T> source(nodes, seq, m_preprocs, m_vec_preprocs);
    VectorX<T> result = m_graph->compute(source, m_outputs.at(idx).first);
    const std::vector<std::string>& labels = m_outputs.at(idx).second;
    std::map<std::string, double> output;
    for (std::size_t iii = 0; iii < labels.size(); iii++) {
      output[labels.at(iii)] = result(iii);
    }
    return output;
  }

  template<typename T>
  VectorMap LightweightGraph<T>::scan(const NodeMap& nodes,
                                     const SeqNodeMap& seq) const {
    return scan(nodes, seq, m_default_output);
  }

  template<typename T>
  VectorMap LightweightGraph<T>::scan(const NodeMap& nodes,
                                     const SeqNodeMap& seq,
                                     const std::string& output) const {
    if (!m_output_indices.count(output)) {
      throw NNEvaluationException("no output node " + output);
    }
    return scan(nodes, seq, m_output_indices.at(output));
  }

  template<typename T>
  VectorMap LightweightGraph<T>::scan(const NodeMap& nodes,
                                     const SeqNodeMap& seq,
                                     std::size_t idx) const {
    LazySource<T> source(nodes, seq, m_preprocs, m_vec_preprocs);
    MatrixX<T> result = m_graph->scan(source, m_outputs.at(idx).first);
    const std::vector<std::string>& labels = m_outputs.at(idx).second;
    std::map<std::string, std::vector<double> > output;
    for (std::size_t iii = 0; iii < labels.size(); iii++) {
      VectorX<T> row = result.row(iii);
      std::vector<double> out_vector(row.data(), row.data() + row.size());
      output[labels.at(iii)] = out_vector;
    }
    return output;
  }

} // namespace generic
} // namespace lwt

#endif
