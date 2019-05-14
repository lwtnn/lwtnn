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
mkdir build

pushd .
cd build
ARGS="-DCMAKE_CXX_STANDARD=${STANDARD-11}"
if [[ ${MINIMAL+x} ]]; then
    ARGS+=" -DBUILTIN_BOOST=TRUE -DBUILTIN_EIGEN=TRUE"
fi
export MAKEFLAGS="-j`nproc` -l`nproc`"
cmake ${ARGS} ..
cmake --build .
ctest --output-on-failure -j`nproc`
popd

if [[ -d bin ]]; then
    echo "removing old bin directory" >&2
    rm -r bin
fi
mv build/bin .

