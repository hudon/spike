Spike
=====

Brain Simulator Parallelization
-------------------------------

[![Build Status](https://travis-ci.org/Hudon/spike.png)](https://travis-ci.org/Hudon/spike)

This project aims to bring performance enhancements to the [nengo project](http://www.nengo.ca/) through parallelization.
The model for concurrency here is to run ensembles concurrently and to split up ensembles to evenly distribute CPU load.
Concurrent ensembles and sub-ensembles can then run on different cores and communicate through pipes, or placed
on different nodes and communicate through a fast transport protocol such as [0MQ](http://www.zeromq.org/).

The main contributors of this project are:
* [Greta Cutulenco](https://github.com/gretac)
* [Robert Elder](https://github.com/robertelder)
* [James Hudon](https://github.com/Hudon)
* [Artem Pasyechnyk](https://github.com/artemip)
* 

Requirements
------------

* numpy == 1.7.x
* scipy == 0.11.0
* theano == 0.6.0rc3
* zmq == 2.2.0


Examples
--------
To run examples:
* Java: get latest Nengo
  [here](http://ctnsrv.uwaterloo.ca:8080/jenkins/job/Nengo/lastSuccessfulBuild/artifact/nengo-latest.zip)
  then run `./nengo-cl examples/java/matrix_multiplication.py`
* numpy: Get python2.x. Get NumPy. Get SciPy. Then, run `python
  examples/numpy/matrix_multiplication.py`
* theano: Get `python-pip`, `python-nose`, `blas`. Then, run `sudo pip install
  Theano`. Then, run `python examples/theano/matrix_multiplication.py`
* core.py: This is a stripped down example (not a real model). It uses numpy,
  so run as `python core.py` once numpy is installed

Theano + 0MQ
------------

Our initial model is to simply run neural ensembles in parallel.
The version in `src/distribute-minimal` does exatly this. It takes the simple system described here
[http://nengo.ca/docs/html/nef_algorithm.html](http://nengo.ca/docs/html/nef_algorithm.html)
and attemps to parallelize it. 

The distributed parallel prototype in `src/distribute-proto` is a second example of
our distribution model, this time we enhance a system that is actually in-use at the Center for Theoretical Neuroscience.
We place one ensemble per process and pipe the output from one neural ensemble to the next. Each ensemble's computation
is compiled from python to C using the [Theano](http://deeplearning.net/software/theano/) library.

Our initial results for these two models show an ideal performance increase: placing two equally-sized ensembles on two
parallel threads of execution increases performance by a factor of two. Placing three ensembles in parallel
increases performance by 3, etc.

Currently, we are working on splitting an ensemble so that we may obtain equally-sized subensembles in a network
that does not have uniformly sized ensembles. We are also running this model over a cluster using 0MQ.




