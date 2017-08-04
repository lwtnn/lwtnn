
# Set the minimum CMake version required to build the project.
cmake_minimum_required( VERSION 3.1 )

# Set the project's name and version.
project( lwtnn VERSION 1.0 )

# Set up the "C++ version" to use.
set( CMAKE_CXX_STANDARD 11 CACHE STRING
   "C++ standard to use for the build" )
set( CMAKE_CXX_EXTENSIONS FALSE CACHE BOOL
   "(Dis)allow using compiler extensions" )

# If the user didn't request a build type explicitly, use an optimised
# build with debug symbols.
if( "${CMAKE_BUILD_TYPE}" STREQUAL "" )
   set( CMAKE_BUILD_TYPE "RelWithDebInfo" )
endif()
message( STATUS "Using build type: ${CMAKE_BUILD_TYPE}" )

# Set the warning flag(s) to use.
set( CMAKE_CXX_FLAGS "-Wall -pedantic" )

# Turn off the usage of RPATH completely:
set( CMAKE_SKIP_RPATH ON )
set( CMAKE_SKIP_BUILD_RPATH ON )
set( CMAKE_SKIP_INSTALL_RPATH ON )

# Set the location of the built libraries/executables inside the build
# directory.
set( CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin )
set( CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib )
set( CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib )

# Make the project's modules visible to CMake.
list( INSERT CMAKE_MODULE_PATH 0 ${CMAKE_SOURCE_DIR}/cmake )

# Find the (necessary) Boost and Eigen headers.
find_package( Boost REQUIRED )
find_package( Eigen3 REQUIRED )

# Public header files for the shared/static library.
set( lib_headers include/lwtnn/Exceptions.hh include/lwtnn/Graph.hh
   include/lwtnn/InputPreprocessor.hh include/lwtnn/LightweightGraph.hh
   include/lwtnn/LightweightNeuralNetwork.hh include/lwtnn/NNLayerConfig.hh
   include/lwtnn/NanReplacer.hh include/lwtnn/Source.hh include/lwtnn/Stack.hh
   include/lwtnn/Stack.hh include/lwtnn/lightweight_network_config.hh
   include/lwtnn/lightweight_nn_streamers.hh include/lwtnn/parse_json.hh )
# Source files for the shared/static library.
set( lib_sources src/Exceptions.cxx src/Graph.cxx src/InputPreprocessor.cxx
   src/LightweightGraph.cxx src/LightweightNeuralNetwork.cxx src/NanReplacer.cxx
   src/Stack.cxx src/lightweight_nn_streamers.cxx src/parse_json.cxx
   src/test_utilities.hh src/test_utilities.cxx )

# Build the shared library.
add_library( lwtnn SHARED ${lib_headers} ${lib_sources} )
target_include_directories( lwtnn
   PUBLIC ${EIGEN3_INCLUDE_DIR}
   $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>
   PRIVATE ${Boost_INCLUDE_DIRS} )
set_property( TARGET lwtnn
   PROPERTY PUBLIC_HEADER ${lib_headers} )

# Build the static library.
add_library( lwtnn-stat ${lib_headers} ${lib_sources} )
target_include_directories( lwtnn-stat
   PUBLIC ${EIGEN3_INCLUDE_DIR}
   $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>
   PRIVATE ${Boost_INCLUDE_DIRS} )
set_property( TARGET lwtnn-stat
   PROPERTY PUBLIC_HEADER ${lib_headers} )

# Install the libraries.
install( TARGETS lwtnn lwtnn-stat
   EXPORT lwtnnTargets
   ARCHIVE DESTINATION lib
   LIBRARY DESTINATION lib
   PUBLIC_HEADER DESTINATION include/lwtnn )

# Helper macro for building the projects' executables.
macro( lwtnn_add_executable name )
   # Declare the executable.
   add_executable( ${name} src/${name}.cxx )
   # Set its properties.
   target_link_libraries( ${name} lwtnn-stat )
   # Install it.
   install( TARGETS ${name}
      EXPORT lwtnnTargets
      RUNTIME DESTINATION bin )
endmacro( lwtnn_add_executable )

# Build/install all executables of the project.
lwtnn_add_executable( lwtnn-benchmark-rnn )
lwtnn_add_executable( lwtnn-dump-config )
lwtnn_add_executable( lwtnn-test-arbitrary-net )
lwtnn_add_executable( lwtnn-test-graph )
lwtnn_add_executable( lwtnn-test-lightweight-graph )
lwtnn_add_executable( lwtnn-test-rnn )

# Install the converter scripts.
install( FILES
   converters/keras2json.py
   converters/agile2json.py
   converters/julian2json.py
   converters/keras_layer_converters_common.py
   converters/keras_v1_layer_converters.py
   converters/keras_v2_layer_converters.py
   converters/kerasfunc2json.py
   PERMISSIONS OWNER_EXECUTE OWNER_READ
   DESTINATION converters )

# Install the CMake description of the project.
install( EXPORT lwtnnTargets
   FILE lwtnnConfig-targets.cmake
   DESTINATION cmake
   NAMESPACE "lwtnn::" )
configure_file( ${CMAKE_SOURCE_DIR}/cmake/lwtnnConfig-version.cmake.in
   ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/lwtnnConfig-version.cmake @ONLY )
install( FILES ${CMAKE_SOURCE_DIR}/cmake/lwtnnConfig.cmake
   ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/lwtnnConfig-version.cmake
   DESTINATION cmake )
