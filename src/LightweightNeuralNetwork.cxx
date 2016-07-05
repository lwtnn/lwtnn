#include "lwtnn/LightweightNeuralNetwork.hh"
#include <Eigen/Dense>

#include <set>

// internal utility functions
namespace {
  using namespace Eigen;
  using namespace lwt;
}
namespace lwt {

  // ______________________________________________________________________
  // Input preprocessor
  InputPreprocessor::InputPreprocessor(const std::vector<Input>& inputs):
    _offsets(inputs.size()),
    _scales(inputs.size())
  {
    size_t in_num = 0;
    for (const auto& input: inputs) {
      _offsets(in_num) = input.offset;
      _scales(in_num) = input.scale;
      _names.push_back(input.name);
      in_num++;
    }
  }
  VectorXd InputPreprocessor::operator()(const ValueMap& in) const {
    VectorXd invec(_names.size());
    size_t input_number = 0;
    for (const auto& in_name: _names) {
      if (!in.count(in_name)) {
        throw NNEvaluationException("can't find input: " + in_name);
      }
      invec(input_number) = in.at(in_name);
      input_number++;
    }
    return (invec + _offsets).cwiseProduct(_scales);
  }

  // ______________________________________________________________________
  // LightweightNeuralNetwork HL wrapper
  LightweightNeuralNetwork::LightweightNeuralNetwork(
    const std::vector<Input>& inputs,
    const std::vector<LayerConfig>& layers,
    const std::vector<std::string>& outputs):
    _stack(inputs.size(), layers),
    _preproc(inputs),
    _outputs(outputs.begin(), outputs.end())
  {
    if (_outputs.size() != _stack.n_outputs()) {
      std::string problem = "internal stack has " +
        std::to_string(_stack.n_outputs()) + " outputs, but " +
        std::to_string(_outputs.size()) + " were given";
      throw NNConfigurationException(problem);
    }
  }

  lwt::ValueMap
  LightweightNeuralNetwork::compute(const lwt::ValueMap& in) const {

    // compute outputs
    auto outvec = _stack.compute(_preproc(in));
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


}
