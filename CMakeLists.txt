cmake_minimum_required (VERSION 3.0)
project(lwtnn)
set(CMAKE_BUILD_TYPE Release)

#Include the headers
set( EIGEN_INC "$ENV{EIGEN_INC}" )
if( NOT EIGEN_INC )
    message( FATAL_ERROR "Please point the environment variable EIGEN_INC
    to the include directory of your eigen3 installation.")
endif()
set( BOOST_INC "$ENV{BOOST_INC}" )
if( NOT BOOST_INC )
    message( FATAL_ERROR "Please point the environment variable BOOST_INC
    to the include directory of your Boost installation.")
endif()
include_directories ( "${EIGEN_INC}" )
include_directories ( "${BOOST_INC}" )
include_directories(include)
 
set(SOURCES
src/Exceptions.cxx
src/Graph.cxx
src/InputPreprocessor.cxx
src/LightweightGraph.cxx
src/LightweightNeuralNetwork.cxx
src/NanReplacer.cxx
src/Stack.cxx
src/lightweight_nn_streamers.cxx
#src/lwtnn-benchmark-rnn.cxx
#src/lwtnn-dump-config.cxx
#src/lwtnn-test-arbitrary-net.cxx
#src/lwtnn-test-graph.cxx
#src/lwtnn-test-rnn.cxx
src/parse_json.cxx)
 
#Generate the shared library from the sources
add_library(lwtnn SHARED ${SOURCES})
 
#add_executable(lwtnn-test-lightweight-graph 
#src/lwtnn-test-lightweight-graph.cxx)

#target_link_libraries(lwtnn-test-lightweight-graph ${SOURCES})




