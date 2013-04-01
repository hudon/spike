import ensemble
import input

from theano import tensor as TT
import theano
import numpy
import random
from multiprocessing import Process, Pipe


# NOTE: ensemble in this class can be either ensemble OR input
# parent_ensemble and sub_index are only used if the node represents a been
# subensemble (ensemble that has split for performance)
class Node:
    def __init__(self, ensemble, process, timer_conn, parent_ensemble=None,
            sub_index=0):
        self.ensemble = ensemble
        self.process = process
        self.timer_conn = timer_conn
        self.parent_ensemble = parent_ensemble
        self.sub_index = sub_index

    def is_subensemble(self):
        return self.parent_ensemble is not None

    def get_parent_name(self):
        return self.parent_ensemble.name

class Network:
    def __init__(self, name, seed=None):
        self.name = name
        self.dt = 0.001
        self.seed = seed
        self.setup = False

        # all the nodes in the network, indexed by name
        self.nodes = {}

        # the list of nodes who have non-theano code that must be run
        # each timestep
        if seed is not None:
            self.random = random.Random()
            self.random.seed(seed)

    # create an ensemble and a specified number of subensembles (num_subs is
    # the number of subensembles)
    # Note that all ensembles are by default arrays of length 1
    def make(self, name, neurons, dimensions, array_count=1, intercept=(-1, 1),
            seed=None, type='lif', encoders=None, num_subs=1):

        if num_subs < 1:
            print >> sys.stderr, "num_subs for ensembles must be greater than 0"
            exit(1)

        # TODO: is this necessary?
        # we need to run the setup again if ensembles are added
        self.setup = False

        if seed is None and self.seed is not None:
            seed = self.random.randrange(0x7fffffff)

        e = ensemble.Ensemble(neurons, dimensions, count = array_count,
            intercept = intercept, dt = self.dt, seed = seed, type=type, 
            encoders=encoders, name=name)

        # if no subensembles, create just the main ensemble process and exit
        if num_subs == 1:
            timer_conn, node_conn = Pipe()
            p = Process(target=e.run, args=(node_conn, ), name=name)
            self.nodes[name] = Node(e, p, timer_conn)
            return

        e_num = 0
        # create the specified number of subensembles
        for encoder, decoder, bias in e.get_subensemble_parts(num_subs):
            subname = e.name + str(e_num)

            e_sub = ensemble.Ensemble(neurons / num_subs, dimensions,
                count=array_count, intercept=intercept, dt=self.dt,
                seed=seed, type=type,
                encoders=encoder,
                is_subensemble=True,
                name=subname,
                decoders=decoder,
                bias=bias)

            # creating a process for each subensemble
            timer_conn, node_conn = Pipe()
            p = Process(target=e_sub.run, args=(node_conn, ), name=subname)
            self.nodes[subname] = Node(e_sub, p, timer_conn, e, e_num)

            e_num += 1

    # wrapper for make function
    def make_array(self, name, neurons, count, dimensions = 1, **args):
        return self.make(name = name, neurons = neurons, dimensions = dimensions,
                array_count = count, **args)

    def make_input(self, name, value, zero_after=None):
        self.add(input.Input(name, value, zero_after=zero_after))

    # add an arbitrary non-theano node (used for Input now, should be used for
    # SimpleNodes when those get implemented
    def add(self, node):
        # we need to run the setup again if ensembles are added
        self.setup = False
        timer_conn, node_conn = Pipe()
        p = Process(target=node.run, args=(node_conn, ), name=node.name)
        self.nodes[node.name] = Node(node, p, timer_conn)

    def get_subnodes(self, parent_name):
        subnodes = []
        for node in self.nodes.values():
            if node.is_subensemble() and node.get_parent_name() is parent_name:
                subnodes.append(node)
        return subnodes

    # This function connects the origin of a node to the accumulator of another
    # node. If the pre is an ensemble that we have split, we must connect all of
    # its subensemble origins to post's accumulators. Same for post (connect all)
    def connect(self, pre, post, transform=None, pstc=0.01,
            func=None, origin_name=None):

        # pre_parent and num_subs are only used if we're connecting to subensembles
        def connect_helper(pre_ens, post_ens, origin_name, pre_index,
                pre_parent=None, num_subs=1):
            # input nodes do not need origins so we can return early.
            # They are identified by the attribute 'value'
            if hasattr(pre_ens, 'value'):
                assert func is None
                next_conn, prev_conn = Pipe()
                pre_ens.add_output(next_conn)
                value_size = len(pre_ens.value)
                post_ens.add_input(prev_conn, pstc, value_size, transform)
                return

            next_conn, prev_conn = Pipe()
            # If there's no func, we use the default 'X' origin created with the
            # ensemble. Otherwise, we create an origin for the func.
            if func is None:
                pre_ens.origin['X'].add_output(next_conn)
                value_size = len(pre_ens.origin['X'].value.eval())
            else:
                if origin_name is None:
                    origin_name = func.__name__

                if origin_name not in pre_ens.origin:
                    decoder = None
                    # If there is a pre_parent, we must use the origin of the
                    # parent ensemble to give the sub_ensemble its decoders.
                    if pre_parent is not None:
                        # take the parent decoder
                        decoder = pre_parent.get_subensemble_decoder(num_subs,
                            origin_name, func)[pre_index]

                    pre_ens.add_origin(origin_name, func, decoder)
                    # print "name " + pre_ens.name + " decoder " + str(pre_ens.origin[origin_name].decoder)

                pre_ens.origin[origin_name].add_output(next_conn)
                value_size = len(pre_ens.origin[origin_name].value.eval())

            post_ens.add_input(prev_conn, pstc, value_size, transform)

        # TODO: is this necessary?
        # we need to run the setup again if ensembles are added
        self.setup = False

        # If pre is not in self.nodes, we can assume that it has been split and
        # that its subensembles are in self.nodes. Same for post.
        if pre in self.nodes:
            pres = [self.nodes[pre]]
        else:
            pres = self.get_subnodes(pre)

        if post in self.nodes:
            posts = [self.nodes[post]]
        else:
            posts = self.get_subnodes(post)

        for pre_node in pres:
            for post_node in posts:
                connect_helper(pre_node.ensemble, post_node.ensemble,
                    origin_name, pre_node.sub_index, pre_node.parent_ensemble, len(pres))

    run_time = 0.0

    # node here is either ensemble OR input
    def make_tick(self, node):
        if hasattr(node, 'make_tick'):
            node.make_tick()

    def run(self, time):
        if not self.setup:
            # NOTE: assuming here that the processes were NOT yet started
            for node in self.nodes.values():
                self.make_tick(node.ensemble)
                node.process.start()
            self.setup = True

        for i in range(int(time / self.dt)):
            t = self.run_time + i * self.dt

            for node in self.nodes.values():
                node.timer_conn.send(t)

            for node in self.nodes.values():
                node.timer_conn.recv()

        self.run_time += time

    # called when the user is all done (otherwise, procs hang :) )
    def clean_up(self):
        # force kill
        for val in self.network_nodes:
            val.process.terminate()
