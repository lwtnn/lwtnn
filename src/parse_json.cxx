#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <cassert>
#include <exception>
#include <iostream>
#include <sstream>
#include <string>

int parse_json()
{
  std::stringstream ss;
  ss << "{ \"root\": { \"values\": [1, 2, 3, 4, 5 ] } }";

  boost::property_tree::ptree pt;
  boost::property_tree::read_json(ss, pt);

  for (const auto& v: pt.get_child("root.values"))
  {
    assert(v.first.empty()); // array elements have no names
    std::cout << v.second.data() << std::endl;
  }
  return 0;
}
