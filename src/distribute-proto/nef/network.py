import ensemble
import input

from theano import tensor as TT
import theano
import numpy
import random
from multiprocessing import Process, Pipe

class Network:
    def __init__(self,name,seed=None):
        self.name=name
        self.dt = 0.001
        self.seed = seed
        self.setup = False

        # all the nodes in the network, indexed by name
        self.nodes = {}
        self.processes = {}

        # the function to call to run the theano portions of the model
        # ahead one timestep
        self.theano_tick = None
        # the list of nodes who have non-theano code that must be run
        # each timestep
        self.tick_nodes = []
        if seed is not None:
            self.random = random.Random()
            self.random.seed(seed)

    # make an ensemble,  Note that all ensembles are actually arrays of length 1
    def make(self, name, neurons, dimensions, array_count=1,
            intercept=(-1, 1), seed=None, type='lif', encoders=None):
        if seed is None:
            if self.seed is not None:
                seed = self.random.randrange(0x7fffffff)

        # just in case the model has been run previously, as adding a new
        # node means we have to rebuild the theano function
        self.theano_tick = None
        e = ensemble.Ensemble(neurons, dimensions, count = array_count,
                intercept = intercept, dt = self.dt, seed = seed,
                type = type, encoders = encoders)
        self.nodes[name] = e

        timer_conn, node_conn = Pipe()
        p = Process(target=e.run, args=(node_conn, ), name=name)

        self.processes[name] = (p, timer_conn)

    def make_array(self, name, neurons, count, dimensions = 1, **args):
        return self.make(name = name, neurons = neurons, dimensions = dimensions,
                array_count = count, **args)

    # create an input
    def make_input(self, name, value, zero_after = None):
        self.add(input.Input(name, value, zero_after=zero_after))

    # add an arbitrary non-theano node (used for Input now, should be used for
    # SimpleNodes when those get implemented
    def add(self, node):
        self.tick_nodes.append(node)
        self.nodes[node.name] = node

        timer_conn, node_conn = Pipe()
        p = Process(target=node.run, args=(node_conn, ), name=node.name)
        self.processes[node.name] = (p, timer_conn)


    def connect(self, pre, post, transform = None, pstc = 0.01, func = None,
            origin_name = None):
        # just in case the model has been run previously, as adding a new
        # node means we have to rebuild the theano function
        self.theano_tick=None

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
                if origin_name is None: origin_name = func.__name__
                if origin_name not in pre.origin:
                    pre.add_origin(origin_name, func)
                value = pre.origin[origin_name].value
            else:
                value = pre.origin['X'].value
        if transform is not None: value = TT.dot(value, transform)

        post.add_filtered_input(value, pstc)

    def connect2(self, pre, post, transform=None, pstc=0.01, func=None,
            origin_name=None):

        self.theano_tick = None

        pre = self.nodes[pre]
        post = self.nodes[post]

        # TODO: consider the transform
        if hasattr(pre, 'value'):
            assert func is None

            next_conn, prev_conn = Pipe()
            pre.add_output(next_conn)
            post.add_input(prev_conn, pstc)
        else:
            next_conn, prev_conn = Pipe()

            if func is not None:
                if origin_name is None:
                    origin_name = func.__name__

                if origin_name not in pre.origin:
                    pre.add_origin(origin_name, func)

                pre.origin[origin_name].add_output(next_conn, transform)
            else:
                pre.origin['X'].add_output(next_conn, transform)

            post.add_input(prev_conn, pstc)

    def make_tick(self):
        updates = {}
        # values() gets all the ensembles added
        for e in self.nodes.values():
            if hasattr(e,'update'):
                updates.update(e.update())
        return theano.function([], [], updates = updates)

    run_time = 0.0
    def run(self, time):
        if not self.setup:
            for e in self.nodes.values():
                if hasattr(e,'make_tick'):
                    e.make_tick()

            for proc, timer_conn in self.processes.values():
                proc.start()

            self.setup = True

        for i in range(int(time / self.dt)):
            t = self.run_time + i * self.dt

            for proc, timer_conn in self.processes.values():
                timer_conn.send(t)

            for proc, timer_conn in self.processes.values():
                timer_conn.recv()

        self.run_time += time

