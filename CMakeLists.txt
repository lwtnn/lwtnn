cmake_minimum_required (VERSION 3.0)
project(lwtnn)
set(GCC_COVERAGE_COMPILE_FLAGS "-O2 -Wall -fPIC -g -std=c++11 -pedantic")
add_definitions(${GCC_COVERAGE_COMPILE_FLAGS})

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

set( CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin )
set( CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib )
 
set(SOURCES
src/Exceptions.cxx
src/Graph.cxx
src/InputPreprocessor.cxx
src/LightweightGraph.cxx
src/LightweightNeuralNetwork.cxx
src/NanReplacer.cxx
src/Stack.cxx
src/lightweight_nn_streamers.cxx
src/parse_json.cxx
src/test_utilities.cxx
src/test_utilities.hh)
 
#Generate the shared library from the sources
add_library(lwtnn SHARED ${SOURCES})


#Executables to be built
set(EXECUTABLES
src/lwtnn-test-lightweight-graph.cxx
src/lwtnn-benchmark-rnn.cxx
src/lwtnn-dump-config.cxx
src/lwtnn-test-arbitrary-net.cxx
src/lwtnn-test-graph.cxx
src/lwtnn-test-rnn.cxx
)

foreach( exec_full ${EXECUTABLES} )
    # I used a simple string replace, to cut off .cpp.
    string( REPLACE "src/" ""  execname_temp ${exec_full} )
    string( REPLACE ".cxx" ""  execname ${execname_temp} )
    add_executable( ${execname} ${exec_full} )
    # Make sure YourLib is linked to each app
    target_link_libraries( ${execname} lwtnn )
endforeach( exec_full ${EXECUTABLES} )

install(TARGETS lwtnn 
        LIBRARY DESTINATION lib
)
install(DIRECTORY include/lwtnn
        DESTINATION include)

install(DIRECTORY converters
        DESTINATION .)
