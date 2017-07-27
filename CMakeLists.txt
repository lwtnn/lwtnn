cmake_minimum_required (VERSION 3.0)
project(lwtnn)
set(GCC_COVERAGE_COMPILE_FLAGS "-O2 -fPIC -g -std=c++11")
add_definitions(${GCC_COVERAGE_COMPILE_FLAGS})

#Include the headers
# We search the standard places. If they don't exist, 
# the user needs to supply the paths.
# i.e. cmake -DEIGEN_INC=<> -DBOOST_INC=<>
set(EIGEN_INC "/usr/local/include/eigen3" CACHE PATH "Path to Eigen headers")
if( NOT EXISTS ${EIGEN_INC} )
    message( FATAL_ERROR "Please add the command -DEIGEN_INC=<path/to/eigen/headers>
    to the cmake configuration.")
endif()

set(BOOST_INC "/usr/local/include" CACHE PATH "Path to Boost headers")
if( NOT EXISTS ${BOOST_INC}/boost )
    message( FATAL_ERROR "Please add the command -DBOOST_INC=<path/to/boost/headers>
    to the cmake configuration.")
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

# Build each executable and link lwtnn
# Also set it's install path when make install is called
foreach( exec_full ${EXECUTABLES} )
    # Use a simple string to remove .cxx and src/.
    string( REPLACE "src/" ""  execname_temp ${exec_full} )
    string( REPLACE ".cxx" ""  execname ${execname_temp} )
    add_executable( ${execname} ${exec_full} )
    # Link lwtnn for each executable
    target_link_libraries( ${execname} lwtnn )
    install(TARGETS ${execname} 
        RUNTIME DESTINATION bin 
    )
endforeach( exec_full ${EXECUTABLES} )

# Install path for liblwtnn.so
install(TARGETS lwtnn 
        LIBRARY DESTINATION lib
)
# Install path for lwtnn headers
install(DIRECTORY include/lwtnn
        DESTINATION include)

# Install path for converter scripts
install(
    FILES 
    converters/keras2json.py  
    converters/agile2json.py
    converters/julian2json.py
    converters/keras_layer_converters_common.py
    converters/keras_v1_layer_converters.py
    converters/keras_v2_layer_converters.py
    converters/kerasfunc2json.py
    PERMISSIONS OWNER_EXECUTE OWNER_READ
    DESTINATION converters
)

