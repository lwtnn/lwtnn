#include "lwtnn/LightweightGraph.hh"
#include "lwtnn/generic/LightweightGraph.hh"
#include "lwtnn/generic/InputPreprocessor.hh"
#include "lwtnn/generic/Graph.hh"
#include "lwtnn/generic/eigen_typedefs.hh"

namespace lwt {
  // ______________________________________________________________________
  // Lightweight Graph

  LightweightGraph::LightweightGraph(const GraphConfig& config,
                                     std::string default_output) :
    m_impl(new generic::LightweightGraph<double>(config, default_output))
  {
  }

  LightweightGraph::~LightweightGraph()
  {
  }

  ValueMap LightweightGraph::compute(const NodeMap& nodes,
                                     const SeqNodeMap& seq) const {
    return m_impl->compute(nodes, seq);
  }

  ValueMap LightweightGraph::compute(const NodeMap& nodes,
                                     const SeqNodeMap& seq,
                                     const std::string& output) const {
    return m_impl->compute(nodes, seq, output);
  }

  VectorMap LightweightGraph::scan(const NodeMap& nodes,
                                   const SeqNodeMap& seq) const {
    return m_impl->scan(nodes, seq);
  }

  VectorMap LightweightGraph::scan(const NodeMap& nodes,
                                   const SeqNodeMap& seq,
                                   const std::string& output) const {
    return m_impl->scan(nodes, seq, output);
  }

} // namespace lwt
