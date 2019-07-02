#include "lwtnn/LightweightGraph.hh"
#include "lwtnn/InputPreprocessor.hh"
#include "lwtnn/Graph.hh"
#include <Eigen/Dense>

namespace {
  using namespace Eigen;
  using namespace lwt;

  typedef LightweightGraph::NodeMap NodeMap;
  typedef InputPreprocessor IP;
  typedef std::vector<std::pair<std::string, IP*> > Preprocs;
  typedef LightweightGraph::SeqNodeMap SeqNodeMap;
  typedef InputVectorPreprocessor IVP;
  typedef std::vector<std::pair<std::string, IVP*> > VecPreprocs;

  // this is used internally to ensure that we only look up map inputs
  // when the network asks for them.
  class LazySource: public ISource
  {
  public:
    LazySource(const NodeMap&, const SeqNodeMap&,
               const Preprocs&, const VecPreprocs&);
    virtual VectorXd at(size_t index) const override;
    virtual MatrixXd matrix_at(size_t index) const override;
  private:
    const NodeMap& m_nodes;
    const SeqNodeMap& m_seqs;
    const Preprocs& m_preprocs;
    const VecPreprocs& m_vec_preprocs;
  };

  LazySource::LazySource(const NodeMap& n, const SeqNodeMap& s,
                         const Preprocs& p, const VecPreprocs& v):
    m_nodes(n), m_seqs(s), m_preprocs(p), m_vec_preprocs(v)
  {
  }
  VectorXd LazySource::at(size_t index) const
  {
    const auto& proc = m_preprocs.at(index);
    if (!m_nodes.count(proc.first)) {
      throw NNEvaluationException("Can't find node " + proc.first);
    }
    const auto& preproc = *proc.second;
    return preproc(m_nodes.at(proc.first));
  }
  MatrixXd LazySource::matrix_at(size_t index) const
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
  // ______________________________________________________________________
  // Lightweight Graph

  typedef LightweightGraph::NodeMap NodeMap;
  LightweightGraph::LightweightGraph(const GraphConfig& config,
                                     std::string default_output):
    m_graph(new Graph(config.nodes, config.layers))
  {
    for (const auto& node: config.inputs) {
      m_preprocs.emplace_back(
        node.name, new InputPreprocessor(node.variables));
    }
    for (const auto& node: config.input_sequences) {
      m_vec_preprocs.emplace_back(
        node.name, new InputVectorPreprocessor(node.variables));
    }
    size_t output_n = 0;
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

  LightweightGraph::~LightweightGraph() {
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

  ValueMap LightweightGraph::compute(const NodeMap& nodes,
                                     const SeqNodeMap& seq) const {
    return compute(nodes, seq, m_default_output);
  }
  ValueMap LightweightGraph::compute(const NodeMap& nodes,
                                     const SeqNodeMap& seq,
                                     const std::string& output) const {
    if (!m_output_indices.count(output)) {
      throw NNEvaluationException("no output node " + output);
    }
    return compute(nodes, seq, m_output_indices.at(output));
  }
  ValueMap LightweightGraph::compute(const NodeMap& nodes,
                                     const SeqNodeMap& seq,
                                     size_t idx) const {
    LazySource source(nodes, seq, m_preprocs, m_vec_preprocs);
    VectorXd result = m_graph->compute(source, m_outputs.at(idx).first);
    const std::vector<std::string>& labels = m_outputs.at(idx).second;
    std::map<std::string, double> output;
    for (size_t iii = 0; iii < labels.size(); iii++) {
      output[labels.at(iii)] = result(iii);
    }
    return output;
  }

  VectorMap LightweightGraph::scan(const NodeMap& nodes,
                                     const SeqNodeMap& seq) const {
    return scan(nodes, seq, m_default_output);
  }
  VectorMap LightweightGraph::scan(const NodeMap& nodes,
                                     const SeqNodeMap& seq,
                                     const std::string& output) const {
    if (!m_output_indices.count(output)) {
      throw NNEvaluationException("no output node " + output);
    }
    return scan(nodes, seq, m_output_indices.at(output));
  }
  VectorMap LightweightGraph::scan(const NodeMap& nodes,
                                     const SeqNodeMap& seq,
                                     size_t idx) const {
    LazySource source(nodes, seq, m_preprocs, m_vec_preprocs);
    MatrixXd result = m_graph->scan(source, m_outputs.at(idx).first);
    const std::vector<std::string>& labels = m_outputs.at(idx).second;
    std::map<std::string, std::vector<double> > output;
    for (size_t iii = 0; iii < labels.size(); iii++) {
      VectorXd row = result.row(iii);
      std::vector<double> out_vector(row.data(), row.data() + row.size());
      output[labels.at(iii)] = out_vector;
    }
    return output;
  }

}
