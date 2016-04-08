#include "lwtnn/LightweightNeuralNetwork.hh"
#include "lwtnn/parse_json.hh"
#include "lwtnn/NanReplacer.hh"

#include <Eigen/Dense>

#include <iostream>
#include <sstream>
#include <string>
#include <fstream>

std::vector<std::string> parse_line(std::string& line) {
  std::stringstream          line_stream(line);
  std::string                cell;

  std::vector<std::string>   result;
  while(line_stream >> cell) {
    result.push_back(cell);
  }
  return result;
}

void usage(const std::string& name) {
  std::cout << "usage: " << name << " <nn config> <labels> <values>\n"
            << "\n"
            << "Both <labels> and <values> should be whitespace separated.\n"
            << "In the case of values, there should be one jet per line\n";
}

int main(int argc, char* argv[]) {
  if (argc != 4) {
    usage(argv[0]);
    exit(1);
  }
  // Read in the configuration.
  std::string in_file_name(argv[1]);
  std::ifstream in_file(in_file_name);
  auto config = lwt::parse_json(in_file);

  // initialize the tagger from the configuration
  lwt::LightweightNeuralNetwork tagger(
    config.inputs, config.layers, config.outputs);
  lwt::NanReplacer replacer(config.defaults, lwt::rep::all);

  // buffer
  std::string val_line;

  // read in input labels and values
  std::ifstream vars_stream( (std::string(argv[2])) );
  std::getline(vars_stream, val_line);
  const auto labels = parse_line(val_line);

  std::ifstream values( (std::string(argv[3])) );
  while (std::getline(values, val_line)) {
    auto val_strings = parse_line(val_line);
    if (val_strings.size() == 0) continue;
    assert(val_strings.size() == labels.size());
    std::map<std::string, double> nn_in;
    for (size_t iii = 0; iii < labels.size(); iii++) {
      nn_in[labels.at(iii)] = std::stof(val_strings.at(iii));
    }
    auto cleaned_inputs = replacer.replace(nn_in);
    for (const auto& pair: cleaned_inputs) {
      std::cout << pair.first << " " << pair.second << std::endl;
    }
    auto out = tagger.compute(cleaned_inputs);
    // look at the outputs
    double sum = 0;
    for (const auto& okey: config.outputs) {
      std::cout << out.at(okey) << " ";
      sum += out.at(okey);
    }
    std::cout << ": " << sum << std::endl;
  }

  return 0;
}
