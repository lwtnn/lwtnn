#ifndef LWTNN_GRAPH_HH
#define LWTNN_GRAPH_HH

#include "lwtnn/generic/Graph.hh"
#include "lwtnn/Source.hh"

namespace lwt {

  using INode = generic::INode<double>;
  using FeedForwardNode = generic::FeedForwardNode<double>;
  using InputNode = generic::InputNode<double>;
  using ConcatenateNode = generic::ConcatenateNode<double>;
  using AddNode = generic::AddNode<double>;
  using ISequenceNode = generic::ISequenceNode<double>;
  using InputSequenceNode = generic::InputSequenceNode<double>;
  using SequenceNode = generic::SequenceNode<double>;
  using TimeDistributedNode = generic::TimeDistributedNode<double>;

  using Graph = generic::Graph<double>;

}


#endif
