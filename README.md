paranengo
=========

Parallelizing Nengo

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
bindings be installed (on top of the `theano` requirements above). It will
also require Python 3.x (but not yet).


