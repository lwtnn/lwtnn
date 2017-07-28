#!/usr/bin/env bash
BOOST_INC=/usr/include/boost
EIGEN_INC=/usr/include/eigen3
mkdir build
pushd .
cd build
cmake .. -DBOOST_INC=${BOOST_INC} -DEIGEN_INC=${EIGEN_INC}
make -j 4
mv bin ..
popd
