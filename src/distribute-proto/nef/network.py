import ensemble
import input

from theano import tensor as TT
import theano
import numpy
import random


class Network:
    def __init__(self, name, seed=None):
        self.name = name
        self.dt = 0.001
        self.seed = seed

        # all the nodes in the network, indexed by name
        self.nodes = {}

        # the function to call to run the theano portions of the model
        # ahead one timestep
        self.theano_tick = None
        # the list of nodes who have non-theano code that must be run
        # each timestep
        self.tick_nodes = []
        if seed is not None:
            self.random = random.Random()
            self.random.seed(seed)

    # make an ensemble,  Note that all ensembles are by default arrays of
    # length 1
    def make(self, name, neurons, dimensions, array_count=1,
             intercept=(-1, 1), seed=None, type='lif', encoders=None):
        if seed is None:
            if self.seed is not None:
                seed = self.random.randrange(0x7fffffff)

        # just in case the model has been run previously, as adding a new
        # node means we have to rebuild the theano function
        self.theano_tick = None
        e = ensemble.Ensemble(neurons, dimensions, count=array_count,
                              intercept=intercept, dt=self.dt, seed=seed,
                              type=type, encoders=encoders)
        self.nodes[name] = e

    def make_array(self, name, neurons, count, dimensions=1, **args):
        return self.make(name=name, neurons=neurons, dimensions=dimensions,
                         array_count=count, **args)

    # create an input
    def make_input(self, name, value, zero_after=None):
        self.add(input.Input(name, value, zero_after=zero_after))

    # add an arbitrary non-theano node (used for Input now, should be used for
    # SimpleNodes when those get implemented
    def add(self, node):
        self.tick_nodes.append(node)
        self.nodes[node.name] = node

    def connect(self, pre, post, transform=None, pstc=0.01, func=None,
                origin_name=None):
        # just in case the model has been run previously, as adding a new
        # node means we have to rebuild the theano function
        self.theano_tick = None

        pre = self.nodes[pre]
        post = self.nodes[post]

        # used for Input objects now, could also be used for SimpleNode
        # origins when they are written
        if hasattr(pre, 'value'):
            assert func is None
            value = pre.value
        else:
          # this else should only be used for ensembles (maybe reorganize this
          # outer if statement to check if it is an ensemble?)
            if func is not None:
                #TODO: better analysis to see if we need to build a new
                # origin (rather than just relying on the name)
                if origin_name is None:
                    origin_name = func.__name__
                if origin_name not in pre.origin:
                    pre.add_origin(origin_name, func)
                value = pre.origin[origin_name].value
            else:
                value = pre.origin['X'].value
        if transform is not None:
            value = TT.dot(value, transform)
        post.add_filtered_input(value, pstc)

    def make_tick(self):
        updates = {}
        # values() gets all the ensembles added
        for e in self.nodes.values():
            if hasattr(e, 'update'):
                updates.update(e.update())
        return theano.function([], [], updates=updates)

    run_time = 0.0

    def run(self, time):
        if self.theano_tick is None:
            self.theano_tick = self.make_tick()
        for i in range(int(time / self.dt)):
            t = self.run_time + i * self.dt
            for node in self.tick_nodes:    # run the non-theano nodes
                node.t = t
                node.tick()
            self.theano_tick()               # run the theano nodes
        self.run_time += time
