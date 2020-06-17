#ifndef LIGHTWEIGHT_GRAPH_HH
#define LIGHTWEIGHT_GRAPH_HH

/* Lightweight Graph

 The lightweightGraph class is a more flexible version of the
 LightweightNeuralNetwork class. This flexibility comes from the
 ability to read from multiple inputs, merge them, and then expose
 multiple outputs.

 For example, a conventional feed-forward network may be structured
 as follows:

  I  <-- simple input vector
  |
  D  <-- dense feed-forward layer
  |
  O  <-- output activation function

 A graph is more flexible, allowing structures like the following:

  I_s  <-- sequential input
   |
  GRU   I_v  <-- simple input vector
     \ /
      M  <-- merge layer
      |
      D  <-- dense layer
     / \
   D2   D3
    |   |
    |   O_c  <-- multiclass output (softmax activation)
    |
   O_r  <-- regression output (linnear output)

 i.e. a graph can combine any number of sequential and "standard"
 rank-1 inputs, and can use the same internal features to infer many
 different attributes from the input pattern.

 Like the LightweightNeuralNetwork, it contains no Eigen code: it
 only serves as a high-level wrapper to convert std::map objects to
 Eigen objects and Eigen objects back to std::maps. For the
 underlying implementation, see Graph.hh. */

#include "lightweight_network_config.hh"

namespace lwt {

  class Graph;
  
  template<typename T> class InputPreprocessorT;
  template<typename T> class InputVectorPreprocessorT;
  
  using InputPreprocessor = InputPreprocessorT<double>;
  using InputVectorPreprocessor = InputVectorPreprocessorT<double>;

  // We currently allow several input types
  // The "ValueMap" is for simple rank-1 inputs
  typedef std::map<std::string, double> ValueMap;
  // The "VectorMap" is for sequence inputs
  typedef std::map<std::string, std::vector<double> > VectorMap;

  // Graph class
  class LightweightGraph
  {
  public:
    // Since a graph has multiple input nodes, we actually call
    typedef std::map<std::string, ValueMap> NodeMap;
    typedef std::map<std::string, VectorMap> SeqNodeMap;

    // In cases where the graph has multiple outputs, we have to
    // define a "default" output, so that calling "compute" with no
    // output specified doesn't lead to ambiguity.
    LightweightGraph(const GraphConfig& config,
                     std::string default_output = "");

    ~LightweightGraph();
    LightweightGraph(LightweightGraph&) = delete;
    LightweightGraph& operator=(LightweightGraph&) = delete;

    // The simpler "compute" function
    ValueMap compute(const NodeMap&, const SeqNodeMap& = {}) const;

    // More complicated version, only needed when you have multiple
    // output nodes and need to specify the non-default ones
    ValueMap compute(const NodeMap&, const SeqNodeMap&,
                     const std::string& output) const;

    // The simpler "scan" function
    VectorMap scan(const NodeMap&, const SeqNodeMap& = {}) const;

    // More complicated version, only needed when you have multiple
    // output nodes and need to specify the non-default ones
    VectorMap scan(const NodeMap&, const SeqNodeMap&,
                   const std::string& output) const;

  private:
    typedef InputPreprocessor IP;
    typedef InputVectorPreprocessor IVP;
    typedef std::vector<std::pair<std::string, IP*> > Preprocs;
    typedef std::vector<std::pair<std::string, IVP*> > VecPreprocs;

    ValueMap compute(const NodeMap&, const SeqNodeMap&, size_t) const;
    VectorMap scan(const NodeMap&, const SeqNodeMap&, size_t) const;
    Graph* m_graph;
    Preprocs m_preprocs;
    VecPreprocs m_vec_preprocs;
    std::vector<std::pair<size_t, std::vector<std::string> > > m_outputs;
    std::map<std::string, size_t> m_output_indices;
    size_t m_default_output;
  };
}

#endif
