Brain Simulator Parallelization
===============================

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

Python 0MQ Version
------------------

The distributed parallel prototype in src/distribute-proto requires 0MQ Python
bindings be installed (on top of the `theano` requirements above). <del>It will
also require Python 3.x (but not yet).</del> It requires Python 2.x because Theano
is not fully 3.x compatible yet

This version takes the simple system described here
[http://nengo.ca/docs/html/nef_algorithm.html](http://nengo.ca/docs/html/nef_algorithm.html) and attemps to parallelize it
using 0MQ. The theano model is also used for reference and might be the second
system we attempt to parallelize when generalizing the system that distributes
the work.


