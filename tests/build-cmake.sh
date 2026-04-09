#!/usr/bin/env bash

if [[ $- == *i* ]]; then
    echo "don't source me!" >&2
    return 1
fi

set -eu

if [[ -d build ]]; then
    echo "removing old build directory" >&2
    rm -r build
fi

TEST_INSTALL=$(mktemp -d)

echo "building from ${PWD}"
tree .

echo "building in build/"

ARGS="-DCMAKE_CXX_STANDARD=${STANDARD-11}"
if [[ ${MINIMAL+x} ]]; then
    ARGS+=" -DBUILTIN_BOOST=TRUE -DBUILTIN_EIGEN=TRUE"
fi
ARGS+=" -DCMAKE_INSTALL_PREFIX=${TEST_INSTALL}"
NPROC=$(nproc 2> /dev/null || gnproc)
export MAKEFLAGS="-j${NPROC} -l${NPROC}"
cmake ${ARGS} -S . -B build
cmake --build build
ctest --output-on-failure --parallel "${NPROC}" -C "${CMAKE_BUILD_TYPE}" -S build
cmake --install build

if [[ -d bin ]]; then
    echo "removing old bin directory" >&2
    rm -r bin
fi
mv build/bin .

# test installation
TEST_BUILD=$(mktemp -d)
TEST_PROJ=${PWD}/tests/install
echo -e "\nbuilding test project in ${TEST_BUILD}, based on ${TEST_PROJ}"
cd $TEST_BUILD
cmake -DCMAKE_PREFIX_PATH=${TEST_INSTALL} -DCMAKE_BUILD_TYPE=Debug $TEST_PROJ
make
