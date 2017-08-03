cmake_minimum_required(VERSION 3.0)
project(lwtnn)
set(GCC_COVERAGE_COMPILE_FLAGS "-O2 -fPIC -g -std=c++11 -Wall -pedantic")
add_definitions(${GCC_COVERAGE_COMPILE_FLAGS})

find_package(Boost REQUIRED)
find_package(Eigen3 QUIET)
if(NOT EIGEN3_FOUND)
  # Fallback to pkg-config
  find_package(PkgConfig)
  pkg_check_modules(EIGEN3 REQUIRED eigen3)
  # pkg-config uses a slightly different name
  set(EIGEN3_INCLUDE_DIR ${EIGEN3_INCLUDE_DIRS})
endif()

include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${BOOST_INCLUDE_DIR})

include_directories(include)

set( CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin )
set( CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib )

#Executables to be built
file(GLOB EXECUTABLES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} src/lwtnn-*.cxx)

file(GLOB SOURCES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} src/*.cxx )

foreach(item ${EXECUTABLES})
  list(REMOVE_ITEM SOURCES ${item})
endforeach(item)

#Generate the shared library from the sources
add_library(lwtnn SHARED ${SOURCES})

#Generate a static libraty for the executables
add_library(lwtnn-stat STATIC ${SOURCES})

# Build each executable and link lwtnn
# Also set it's install path when make install is called
foreach(exec_full ${EXECUTABLES})
    # Use a simple string to remove .cxx and src/.
    string( REPLACE "src/" ""  execname_temp ${exec_full} )
    string( REPLACE ".cxx" ""  execname ${execname_temp} )
    add_executable( ${execname} ${exec_full})
    # Link lwtnn for each executable
    target_link_libraries( ${execname} lwtnn-stat)
    install(TARGETS ${execname}
        RUNTIME DESTINATION bin
    )
endforeach(exec_full)

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

