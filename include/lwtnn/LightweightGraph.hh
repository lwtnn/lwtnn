#ifndef LIGHTWEIGHT_GRAPH_HH
#define LIGHTWEIGHT_GRAPH_HH

#include "lightweight_network_config.hh"

namespace lwt {

  class Graph;
  class InputPreprocessor;

  typedef std::map<std::string, double> ValueMap;

  // Graph version
  class LightweightGraph
  {
  public:
    typedef std::map<std::string, ValueMap> NodeMap;
    LightweightGraph(const GraphConfig& config,
                     std::string default_output = "");
    ~LightweightGraph();
    LightweightGraph(LightweightGraph&) = delete;
    LightweightGraph& operator=(LightweightGraph&) = delete;

    ValueMap compute(const NodeMap&) const;
    ValueMap compute(const NodeMap&, const std::string& output) const;
  private:
    ValueMap compute(const NodeMap&, size_t) const;
    Graph* m_graph;
    std::vector<std::pair<std::string, InputPreprocessor*> > m_preprocs;
    std::vector<std::pair<size_t, std::vector<std::string> > > m_outputs;
    std::map<std::string, size_t> m_output_indices;
    size_t m_default_output;
  };
}

#endif
