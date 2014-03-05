Spike
=====

Brain Simulator Parallelization
-------------------------------

[![Build Status](https://travis-ci.org/Hudon/spike.png)](https://travis-ci.org/Hudon/spike)

This project aims to bring performance enhancements to the [nengo project](http://www.nengo.ca/) through parallelization.
The model for concurrency here is to run ensembles concurrently and to split up ensembles to evenly distribute CPU load.
Concurrent ensembles and sub-ensembles can then run on different cores and communicate through pipes, or placed
on different nodes and communicate through a fast transport protocol such as [0MQ](http://www.zeromq.org/).

Features of this project include

*  Ability to run ensembles in different processes (and distribute them to other hosts).
*  Ability to split ensemble computation into different processes (and distribute them to other hosts)
*  A simple unit test suite that can be envoked by doing 'make test' in the root directory
*  An interface that is compatible with the previous nengo version
*  Scripts that automate the installation of a headless (no user interface) virtual machine that can run distributed simulation
*  A script that can be used to create an AWS EC2 cluster containing an arbitrary number of instances that runs the distributed simulation
*  Faster performance for larger models.

The main contributors of this project are:
* [Greta Cutulenco](https://github.com/gretac)
* [Robert Elder](https://github.com/robertelder)
* [James Hudon](https://github.com/Hudon)
* [Artem Pasyechnyk](https://github.com/artemip)

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


Running Spike in a Virtual Machine
------------
When the simulated was first distributed, it was done inside a virtual machine in order to be sure that all software versions were identical.
The virtual machine setup was automated in the commands available in 'spike/vm-setup/vm-create-steps'.
You can run these commands with the command-harness script located in that directory.

The virtual machine uses an arch linux image: archlinux-2013.11.01-dual.iso

Because the virtual machines used for the distribution were unable to acquire their own IP address, they used NAT.  This was problematic, because it required that any servers offered by the virtual machine require that port forwarding rules be added so that machines from the external world can interact with the services offered in the virtual machine.  The VM setup script creates port forwarding rules for ports 8000-9000.  Additionally, it was necessary to change the ephemeral port range of the arch installation the VMs used, so that the ephemeral port range could be port forwarded too.  This was necessary because responses from incomming connections are sent back to the client's ephemeral port.  The new ephemeral ports are 10000-10100 which only provides about 100 ephemeral ports.  This works for now, but it might become a problem in the future.  After the virtual machine has been configured, it is booted in headless mode.  We use the keyboardputscancode virtualbox API function to send keystrokes to the virtual machine, and through this method we can pass the start menu and start the ssh daemon.  After this point, we can directly ssh into the virtual machine from the host machine.  At this point we install all the necessary dependencies like theano and scipy that we require for the simulation.


Running Spike in on AWS
------------

Spike can be run on the ec2 cloud.  This can be done with the scripts located at spike/ec2-distribution.

Creating a cluster:
./ec2-spike create-cluster [number of instances]

Deleting a cluster:
./ec2-spike delete-cluster


This script was written for GNU bash, version 4.2.25(1)-release (x86_64-pc-linux-gnu)
Modification may be necessary if you are using a different shell version.

The only pre-requisite for this script is that you set up your shell to work with the AWS CLI:
http://aws.amazon.com/cli/
When you set up the aws CLI you will specify your AWS credentials so that a call to something like 'aws ec2 describe-instances' will succeed in your shell

This script is designed to make it easy to launch a number of ec2 instances and configure them to run
as a single distributed system that runs the spike simulations.
You can envoke this script by specifying 'create-cluster' or 'delete-cluster' and a number for the number of nodes the cluster should have.
After creating the cluster, information about the instances is stored in files so that we can access it later when we decide to terminate the instance.

If you want a example that does a full simulation you can use the example.sh script.

The install instructions for each node are in  ec2-node-install.sh

The kill-daemon.sh script is used to kill the daemon on each of the nodes.  This is useful when re-starting a simulation.

Monitoring you simulation
------------

You can monitor the resource performance of all nodes in your simulation.  This will likely have require some changes for your specific use case.

This can be done with the script in spike/vm-setup/monitor.sh.  This script will need to be modified to include your host names.
