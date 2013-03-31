import ensemble
import input

from theano import tensor as TT
import theano
import numpy
import random
from multiprocessing import Process, Pipe


class Network:
    def __init__(self, name, seed=None):
        self.name = name
        self.dt = 0.001
        self.seed = seed
        self.setup = False

        # all the nodes in the network, indexed by name
        self.nodes = {}
        self.processes = {}
        self.subensembles = {}

        # the list of nodes who have non-theano code that must be run
        # each timestep
        self.tick_nodes = []
        if seed is not None:
            self.random = random.Random()
            self.random.seed(seed)

    # make an ensemble,  Note that all ensembles are by default arrays of
    # length 1
    def make(self, name, neurons, dimensions, array_count=1,
             intercept=(-1, 1), seed=None, type='lif', num_splits=1):

        if num_splits < 1:
            print >> sys.stderr, "num_splits for ensembles must be greater than 0"
            exit(1)

        # we need to run the setup again if ensembles are added
        self.setup = False

        if seed is None:
            if self.seed is not None:
                seed = self.random.randrange(0x7fffffff)

        e = ensemble.Ensemble(neurons, dimensions, count = array_count,
            intercept = intercept, dt = self.dt, seed = seed, type=type, 
            name=name)

        if num_splits > 1:
            encoders = e.encoders
            decoders = e.decoders
            bias = e.bias

            encoder_length = len(encoders)
            decoder_length = len(decoders)
            bias_length = len(bias)

            for e_num in range(1, num_splits + 1):
                subname = e.name + str(e_num)

                e_size = encoder_length / num_splits
                d_size = decoder_length / num_splits
                b_size = bias_length / num_splits

                e1 = ensemble.Ensemble(neurons / num_splits, dimensions,
                    count = array_count, intercept = intercept, dt = self.dt,
                    seed = seed, type=type,
                    encoders=encoders[e_size * (e_num - 1):e_size * e_num],
                    override=True,
                    name=subname,
                    decoders=decoders[d_size * (e_num - 1):d_size * e_num],
                    bias=bias[b_size * (e_num - ):b_size * e_num])

                self.nodes[subname] = e1
                self.subensembles[name] = subname

                # creating a process for EACH subensemble
                timer_conn, node_conn = Pipe()
                p = Process(target=e1.run, args=(node_conn, ), name=subname)
                self.processes[subname] = (p, timer_conn)

        else:
            # if no subensembles, create just the main ensemble process
            timer_conn, node_conn = Pipe()
            p = Process(target=e.run, args=(node_conn, ), name=subname)
            self.processes[subname] = (p, timer_conn)

    # wrapper for make function
    def make_array(self, name, neurons, count, dimensions = 1, **args):
        return self.make(name = name, neurons = neurons, dimensions = dimensions,
                array_count = count, **args)

    # create an input
    def make_input(self, name, value, zero_after=None):
        self.add(input.Input(name, value, zero_after=zero_after))

    # add an arbitrary non-theano node (used for Input now, should be used for
    # SimpleNodes when those get implemented
    def add(self, node):
        # we need to run the setup again if ensembles are added
        self.setup = False

        self.tick_nodes.append(node)
        self.nodes[node.name] = node

        timer_conn, node_conn = Pipe()
        p = Process(target=node.run, args=(node_conn, ), name=node.name)
        self.processes[node.name] = (p, timer_conn)

    def connect(self, pre, post, transform=None, pstc=0.01, func=None,
            origin_name=None):
        # we need to run the setup again if ensembles are added
        self.setup = False

        pre = self.nodes[pre]
        post = self.nodes[post]

        if hasattr(pre, 'value'):
            assert func is None
            next_conn, prev_conn = Pipe()
            pre.add_output(next_conn)
            value_size = len(pre.value)
        else:
            next_conn, prev_conn = Pipe()
            if func is not None:
                if origin_name is None:
                    origin_name = func.__name__
                if origin_name not in pre.origin:
                    pre.add_origin(origin_name, func)

                pre.origin[origin_name].add_output(next_conn)
                value_size = len(pre.origin[origin_name].value.eval())
            else:
                pre.origin['X'].add_output(next_conn)
                value_size = len(pre.origin['X'].value.eval())
        post.add_input(prev_conn, pstc, value_size, transform)

    run_time = 0.0

    def run(self, time):
        if not self.setup:
            for e in self.nodes.values():
                if hasattr(e, 'make_tick'):
                    e.make_tick()

            for proc, timer_conn in self.processes.values():
                if not proc.is_alive():
                    proc.start()
            self.setup = True

        for i in range(int(time / self.dt)):
            t = self.run_time + i * self.dt

            for proc, timer_conn in self.processes.values():
                timer_conn.send(t)

            for proc, timer_conn in self.processes.values():
                timer_conn.recv()

        self.run_time += time

    # called when the user is all done (otherwise, procs hang :) )
    def clean_up(self):
        # force kill
        for proc, timer_conn in self.processes.values():
            proc.terminate()
