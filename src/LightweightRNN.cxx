// Lightweight Recurrent NN
//
//basic code for forward pass computation of recurrent NN structures,
// like LSTM, useful for processing time series / sequence data.  goal
// to be able to evaluate Keras (keras.io) models in c++ in
// lightweight way
//
// Author: Michael Kagan <mkagan@cern.ch>


#include "lwtnn/LightweightRNN.hh"
#include "lwtnn/Stack.hh"


namespace lwt {


  // ______________________________________________________________________
  // Input vector preprocessor
  InputVectorPreprocessor::InputVectorPreprocessor(
    const std::vector<Input>& inputs):
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
    // require at least one input at configuration, since we require
    // at least one for evaluation
    if (in_num == 0) {
      throw NNConfigurationException("need at least one input");
    }
  }
  MatrixXd InputVectorPreprocessor::operator()(const VectorMap& in) const {
    using namespace Eigen;
    if (in.size() == 0) {
      throw NNEvaluationException("Empty input map");
    }
    size_t n_cols = in.begin()->second.size();
    MatrixXd inmat(_names.size(), n_cols);
    size_t in_num = 0;
    for (const auto& in_name: _names) {
      if (!in.count(in_name)) {
        throw NNEvaluationException("can't find input: " + in_name);
      }
      const auto& invec = in.at(in_name);
      if (invec.size() == 0) {
        throw NNEvaluationException("Input vector of zero length");
      }
      if (invec.size() != n_cols) {
        throw NNEvaluationException("Input vector size mismatch");
      }
      inmat.row(in_num) = Map<const VectorXd>(invec.data(), invec.size());
      in_num++;
    }
    return _scales.asDiagonal() * (inmat.colwise() + _offsets);
  }

  // ______________________________________________________________________
  // LightweightRNN

  LightweightRNN::LightweightRNN(const std::vector<Input>& inputs,
                                 const std::vector<LayerConfig>& layers,
                                 const std::vector<std::string>& outputs):
    _stack(inputs.size(), layers),
    _preproc(inputs),
    _vec_preproc(inputs),
    _outputs(outputs.begin(), outputs.end()),
    _n_inputs(inputs.size())
  {
    if (_outputs.size() != _stack.n_outputs()) {
      throw NNConfigurationException(
        "Mismatch between NN output dimensions and output labels");
    }
  }


  ValueMap LightweightRNN::reduce(const std::vector<ValueMap>& in) const {
    MatrixXd inputs(_n_inputs, in.size());
    for (size_t iii = 0; iii < in.size(); iii++) {
      inputs.col(iii) = _preproc(in.at(iii));
    }
    auto outvec = _stack.reduce(inputs);
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
    auto outvec = _stack.reduce(_vec_preproc(in));
    ValueMap out;
    const auto n_rows = static_cast<size_t>(outvec.rows());
    for (size_t iii = 0; iii < n_rows; iii++) {
      out.emplace(_outputs.at(iii), outvec(iii));
    }
    return out;
  }

}

// ________________________________________________________________________
// convenience functions

