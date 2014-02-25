#! /bin/bash
set -e

#  This script will install everything that is needed on a specific ec2 host to run the spike unit tests
sudo apt-get update
sudo apt-get install python-pip python-dev build-essential -y
sudo pip install quantities
sudo apt-get install -y uuid-dev
wget http://download.zeromq.org/zeromq-2.2.0.tar.gz && tar xf zeromq-2.2.0.tar.gz && cd zeromq-2.2.0 && ./configure && make && sudo make install && cd ..
sudo pip install pyzmq
sudo apt-get install libblas-dev gfortran -y
sudo apt-get install -qq python-numpy python-scipy -y
sudo pip install Theano
