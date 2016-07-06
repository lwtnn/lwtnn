#include "lwtnn/LightweightNeuralNetwork.hh"
#include "lwtnn/Stack.hh"
#include <Eigen/Dense>

#include <set>

// internal utility functions
namespace {
  using namespace Eigen;
  using namespace lwt;
}
namespace lwt {

  // ______________________________________________________________________
  // LightweightNeuralNetwork HL wrapper
  LightweightNeuralNetwork::LightweightNeuralNetwork(
    const std::vector<Input>& inputs,
    const std::vector<LayerConfig>& layers,
    const std::vector<std::string>& outputs):
    _stack(new Stack(inputs.size(), layers)),
    _preproc(new InputPreprocessor(inputs)),
    _outputs(outputs.begin(), outputs.end())
  {
    if (_outputs.size() != _stack->n_outputs()) {
      std::string problem = "internal stack has " +
        std::to_string(_stack->n_outputs()) + " outputs, but " +
        std::to_string(_outputs.size()) + " were given";
      throw NNConfigurationException(problem);
    }
  }

  LightweightNeuralNetwork::~LightweightNeuralNetwork() {
    delete _stack;
    _stack = 0;
    delete _preproc;
    _preproc = 0;
  }

  lwt::ValueMap
  LightweightNeuralNetwork::compute(const lwt::ValueMap& in) const {

    // compute outputs
    const auto& preproc = *_preproc;
    auto outvec = _stack->compute(preproc(in));
    assert(outvec.rows() > 0);
    auto out_size = static_cast<size_t>(outvec.rows());
    assert(out_size == _outputs.size());

    // build and return output map
    lwt::ValueMap out_map;
    for (size_t out_n = 0; out_n < out_size; out_n++) {
      out_map.emplace(_outputs.at(out_n), outvec(out_n));
    }
    return out_map;
  }

  // ______________________________________________________________________
  // LightweightRNN

  LightweightRNN::LightweightRNN(const std::vector<Input>& inputs,
                                 const std::vector<LayerConfig>& layers,
                                 const std::vector<std::string>& outputs):
    _stack(new RecurrentStack(inputs.size(), layers)),
    _preproc(new InputPreprocessor(inputs)),
    _vec_preproc(new InputVectorPreprocessor(inputs)),
    _outputs(outputs.begin(), outputs.end()),
    _n_inputs(inputs.size())
  {
    if (_outputs.size() != _stack->n_outputs()) {
      throw NNConfigurationException(
        "Mismatch between NN output dimensions and output labels");
    }
  }
  LightweightRNN::~LightweightRNN() {
    delete _stack;
    delete _preproc;
    delete _vec_preproc;
  }

  ValueMap LightweightRNN::reduce(const std::vector<ValueMap>& in) const {
    const auto& preproc = *_preproc;
    MatrixXd inputs(_n_inputs, in.size());
    for (size_t iii = 0; iii < in.size(); iii++) {
      inputs.col(iii) = preproc(in.at(iii));
    }
    auto outvec = _stack->reduce(inputs);
    ValueMap out;
    const auto n_rows = static_cast<size_t>(outvec.rows());
    for (size_t iii = 0; iii < n_rows; iii++) {
      out.emplace(_outputs.at(iii), outvec(iii));
    }
    return out;
  }

  // this version should be slightly faster since it only has to sort
  // the inputs once
  ValueMap LightweightRNN::reduce(const VectorMap& in) const {
    const auto& preproc = *_vec_preproc;
    auto outvec = _stack->reduce(preproc(in));
    ValueMap out;
    const auto n_rows = static_cast<size_t>(outvec.rows());
    for (size_t iii = 0; iii < n_rows; iii++) {
      out.emplace(_outputs.at(iii), outvec(iii));
    }
    return out;
  }


}
