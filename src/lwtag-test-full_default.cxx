#include "LightweightNeuralNetwork.hh"
#include "parse_json.hh"

#include <Eigen/Dense>

#include <iostream>
#include <string>
#include <sstream>
#include <fstream>

int main(int argc, char* argv[]) {
  // Read in the configurationusing an istringstream.
  std::ifstream weight_file("/Users/marie/Work/GitHub/lw-client/data/test-nn.json");
  std::string line_str;
  std::string weight_file_str;
  while (std::getline(weight_file, line_str))
    {
      weight_file_str += line_str+ ' ' ;
    }
  std::istringstream weight_file_sstream(weight_file_str);
  lwt::JSONConfig config = lwt::parse_json(weight_file_sstream);
  std::map<std::string,double> m_map_defaults = lwt::get_defaults_from_json(weight_file_sstream);

  // initialize the tagger from the configuration
  lwt::LightweightNeuralNetwork tagger(config.inputs, config.layers, config.outputs);

  // build some dummy inputs and feed them to the tagger
  lwt::LightweightNeuralNetwork::ValueMap input{
    {"in1", 1}, {"in2", 2}, {"in3", NAN}, {"in4", 4} };
  
  std::map<std::string,double> corrected_input = input; // create copy of input map to ensure nothing is missing

  // check inputs and replace defaults           
  for (const auto in:input) {
    for (const auto in_default:m_map_defaults) {
      if (std::isnan(in.second) && in.first==in_default.first)
	{
	  std::cout << "Default value found:/nDefault:" << in_default.first << " = " << in_default.second << std::endl;
	  std::cout << "Input: "<<in.first<<" = " << in.second << std::endl;
	  corrected_input.at(in.first) = in_default.second;
	  std::cout << "Changed input variable value to default value." << std::endl;
      	  continue;
	}
    }
  }

  // look at the corrected inputs
  std::cout<< "Checking corrected input:"<<std::endl;
  for(const auto ci:corrected_input)
    {
      std::cout << ci.first << " = " << ci.second << std::endl;
    }

  auto out = tagger.compute(corrected_input);

  // look at the outputs
  for (const auto& op: out) {
    std::cout << op.first << " " << op.second << std::endl;
  }

  return 0;
}
