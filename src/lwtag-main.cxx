#include "tools.hh"

#include "LWTagger.hh"

#include <Eigen/Dense>

#include <iostream>

// #include <cstdio>
// #include <cstdlib>
int main(int argc, char* argv[]) {
  // int in = 0;
  // if (argc > 1) in = atoi(argv[1]);
  // int testint = testfunc(in);
  // printf("bonjour et %i\n", testint);

  lwt::Stack stack;

  Eigen::VectorXd input(4);
  input << 1, 2, 3, 4;
  std::cout << stack.compute(input) << std::endl;
  return 0;
}
