cmake_minimum_required (VERSION 3.0)
project(lwtnn)
set(GCC_COVERAGE_COMPILE_FLAGS "-O2 -Wall -fPIC -g -std=c++11 -pedantic")
add_definitions(${GCC_COVERAGE_COMPILE_FLAGS})

find_package(Eigen3 REQUIRED)
find_package(Boost REQUIRED)

include_directories (${EIGEN3_INCLUDE_DIR})
include_directories (${BOOST_INCLUDE_DIR})
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

# Build each exectuable and link lwtnn
# Also set it's install path when make install is called
foreach( exec_full ${EXECUTABLES} )
    # I used a simple string replace, to cut off .cpp.
    string( REPLACE "src/" ""  execname_temp ${exec_full} )
    string( REPLACE ".cxx" ""  execname ${execname_temp} )
    add_executable( ${execname} ${exec_full} )
    # Make sure YourLib is linked to each app
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

