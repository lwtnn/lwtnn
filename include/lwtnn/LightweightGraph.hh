#ifndef LIGHTWEIGHT_GRAPH_HH
#define LIGHTWEIGHT_GRAPH_HH

#include "lightweight_network_config.hh"

namespace lwt {

  class Graph;
  class InputPreprocessor;
  class InputVectorPreprocessor;

  typedef std::map<std::string, double> ValueMap;
  typedef std::map<std::string, std::vector<double> > VectorMap;

  // Graph version
  class LightweightGraph
  {
  public:
    typedef std::map<std::string, ValueMap> NodeMap;
    typedef std::map<std::string, VectorMap> SeqNodeMap;
    LightweightGraph(const GraphConfig& config,
                     std::string default_output = "");
    ~LightweightGraph();
    LightweightGraph(LightweightGraph&) = delete;
    LightweightGraph& operator=(LightweightGraph&) = delete;

    ValueMap compute(const NodeMap&, const SeqNodeMap& = {}) const;
    ValueMap compute(const NodeMap&, const SeqNodeMap&,
                     const std::string& output) const;
  private:
    typedef InputPreprocessor IP;
    typedef InputVectorPreprocessor IVP;
    typedef std::vector<std::pair<std::string, IP*> > Preprocs;
    typedef std::vector<std::pair<std::string, IVP*> > VecPreprocs;

    ValueMap compute(const NodeMap&, const SeqNodeMap&, size_t) const;
    Graph* m_graph;
    Preprocs m_preprocs;
    VecPreprocs m_vec_preprocs;
    std::vector<std::pair<size_t, std::vector<std::string> > > m_outputs;
    std::map<std::string, size_t> m_output_indices;
    size_t m_default_output;
  };
}

#endif
