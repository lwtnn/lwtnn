#!/usr/bin/env bash
export BOOST_INC=/usr/include/boost
export EIGEN_INC=/usr/include/eigen3
mkdir build
pushd .
cd build
cmake ..
make -j 4
mv bin ..
popd
