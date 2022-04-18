cmake_minimum_required(VERSION 3.12...3.20)
project(TestLwtnn)

set( CMAKE_CXX_STANDARD 11 CACHE STRING
  "C++ standard to use for the build" )

find_package( lwtnn REQUIRED )
add_executable(test-lwtnn test-lwtnn.cxx )
target_link_libraries( test-lwtnn lwtnn::lwtnn )
