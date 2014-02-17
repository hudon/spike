#! /bin/bash
set -e

sudo apt-get update
sudo apt-get install python-pip python-dev build-essential -y
sudo pip install quantities
sudo apt-get install -y uuid-dev
wget http://download.zeromq.org/zeromq-2.2.0.tar.gz && tar xf zeromq-2.2.0.tar.gz && cd zeromq-2.2.0 && ./configure && make && sudo make install && cd ..
sudo pip install pyzmq
sudo apt-get install libblas-dev gfortran -y
sudo apt-get install -qq python-numpy python-scipy -y
sudo pip install Theano
