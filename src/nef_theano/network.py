import random
from _collections import OrderedDict
import quantities

import theano
from theano import tensor as TT
import numpy as np

from . import ensemble
from . import simplenode
from . import probe
from . import origin
from . import input
from . import subnetwork
from . import connection

from multiprocessing import Process
import zmq
from . import zmq_utils

class Network(object):
    def __init__(self, name, seed=None, fixed_seed=None, dt=.001):
        """Wraps an NEF network with a set of helper functions
        for simplifying the creation of NEF models.

        :param string name:
            create and wrap a new Network with the given name.
        :param int seed:
            random number seed to use for creating ensembles.
            This one seed is used only to start the
            random generation process, so each neural group
            created will be different.

        """
        self.name = name
        self.dt = dt
        self.run_time = 0.0    
        self.seed = seed
        self.fixed_seed = fixed_seed

        # the input and spiking ensemble nodes in the network
        self.nodes = {}
        self.processes = []

        self.zmq_context = zmq.Context()
        self.setup = False

        self.random = random.Random()
        if seed is not None:
            self.random.seed(seed)

    def add(self, node):
        """Add an arbitrary non-theano node to the network.

        Used for inputs, SimpleNodes, and Probes. These nodes will be
        added to the Theano graph if the node has an "update()" function,
        but will also be triggered explicitly at every tick
        via the node's `theano_tick()` function.

        :param Node node: the node to add to this network

        """
        self.nodes[node.name] = node

        ticker_socket, node_socket = \
            zmq_utils.create_socket_defs_reqrep("ticker", node.name)
        p = Process(target=node.run, args=(node_socket,), name=node.name)
        self.processes.append((p,
            ticker_socket.create_socket(self.zmq_context),))

    def connect(self, pre, post, transform=None, weight=1,
                index_pre=None, index_post=None, pstc=0.01, 
                func=None):
        """Connect two nodes in the network.

        Note: cannot specify (transform) AND any of
        (weight, index_pre, index_post).

        *pre* and *post* can be strings giving the names of the nodes,
        or they can be the nodes themselves (Inputs and Ensembles are
        supported). They can also be actual Origins or Terminations,
        or any combination of the above. 

        If transform is not None, it is used as the transformation matrix
        for the new termination. You can also use *weight*, *index_pre*,
        and *index_post* to define a transformation matrix instead.
        *weight* gives the value, and *index_pre* and *index_post*
        identify which dimensions to connect.

        transform can be of several sizes:

        - post.dimensions * pre.dimensions:
          Specify where decoded signal dimensions project
        - post.neurons * pre.dimensions:
          Overwrites post encoders, i.e. inhibitory connections
        - post.neurons * pre.neurons:
          Fully specify the connection weight matrix 

        If *func* is not None, a new Origin will be created on the
        pre-synaptic ensemble that will compute the provided function.
        The name of this origin will be taken from the name of
        the function, or *origin_name*, if provided. If an
        origin with that name already exists, the existing origin
        will be used rather than creating a new one.

        :param string pre: Name of the node to connect from.
        :param string post: Name of the node to connect to.
        :param float pstc:
            post-synaptic time constant for the neurotransmitter/receptor
            on this connection
        :param transform:
            The linear transfom matrix to apply across the connection.
            If *transform* is T and *pre* represents ``x``,
            then the connection will cause *post* to represent ``Tx``.
            Should be an N by M array, where N is the dimensionality
            of *post* and M is the dimensionality of *pre*.
        :type transform: array of floats
        :param index_pre:
            The indexes of the pre-synaptic dimensions to use.
            Ignored if *transform* is not None.
            See :func:`connection.compute_transform()`
        :param float weight:
            Scaling factor for a transformation defined with
            *index_pre* and *index_post*.
            Ignored if *transform* is not None.
            See :func:`connection.compute_transform()`
        :type index_pre: List of integers or a single integer
        :param index_post:
            The indexes of the post-synaptic dimensions to use.
            Ignored if *transform* is not None.
            See :func:`connection.compute_transform()`
        :type index_post: List of integers or a single integer 
        :param function func:
            Function to be computed by this connection.
            If None, computes ``f(x)=x``.
            The function takes a single parameter ``x``, which is
            the current value of the *pre* ensemble, and must return
            either a float or an array of floats.
        :param string origin_name:
            Name of the origin to check for / create to compute
            the given function.
            Ignored if func is None. If an origin with this name already
            exists, the existing origin is used
            instead of creating a new one.

        """

        # get post Node object from node dictionary
        post = self.get_object(post)

        # get pre Node object from node dictionary
        pre_name = pre
        pre = self.get_object(pre)
        # get the origin from the pre Node, CREATE one if does not exist
        pre_origin = self.get_origin(pre_name, func)
        pre_output = pre_origin.decoded_output
        dim_pre = pre_origin.dimensions 

        if transform is not None: 

            # there are 3 cases
            # 1) pre = decoded, post = decoded
            #     - in this case, transform will be 
            #                       (post.dimensions x pre.origin.dimensions)
            #     - decoded_input will be (post.array_size x post.dimensions)
            # 2) pre = decoded, post = encoded
            #     - in this case, transform will be size 
            #         (post.array_size x post.neurons x pre.origin.dimensions)
            #     - encoded_input will be (post.array_size x post.neurons_num)
            # 3) pre = encoded, post = encoded
            #     - in this case, transform will be (post.array_size x 
            #             post.neurons_num x pre.array_size x pre.neurons_num)
            #     - encoded_input will be (post.array_size x post.neurons_num)

            # make sure contradicting things aren't simultaneously specified
            assert ((weight == 1) and (index_pre is None)
                    and (index_post is None))

            transform = np.array(transform)

            # check to see if post side is an encoded connection, case 2 or 3
            #TODO: a better check for this
            if transform.shape[0] != post.dimensions * post.array_size or len(transform.shape) > 2:

                raise Exception("ERROR", "Case 2 and 3 should NOT be reached.")

                if transform.shape[0] == post.array_size * post.neurons_num:
                    transform = transform.reshape(
                                      [post.array_size, post.neurons_num] +\
                                                list(transform.shape[1:]))

                if len(transform.shape) == 2: # repeat array_size times
                    transform = np.tile(transform, (post.array_size, 1, 1))

                # check for pre side encoded connection (case 3)
                if len(transform.shape) > 3 or \
                       transform.shape[2] == pre.array_size * pre.neurons_num:

                    if transform.shape[2] == pre.array_size * pre.neurons_num: 
                        transform = transform.reshape(
                                        [post.array_size, post.neurons_num,  
                                              pre.array_size, pre.neurons_num])
                    assert transform.shape == \
                            (post.array_size, post.neurons_num, 
                             pre.array_size, pre.neurons_num)

                    print 'setting pre_output=spikes'

                    # get spiking output from pre population
                    pre_output = pre.neurons.output 

                    case1 = connection.Case1(
                        (post.array_size, post.neurons_num,
                         pre.array_size, pre.neurons_num))
                    encoded_output = case1(transform, pre_output)

                    # pass in the pre population encoded output function
                    # to the post population, connecting them for theano
                    post.add_termination(name=pre_name, pstc=pstc, 
                        encoded_input=encoded_output)

                    return

                else: # otherwise we're in case 2 (pre is decoded)
                    raise Exception("ERROR", "Case 2 and 3 should NOT be reached.")
                    assert transform.shape ==  \
                               (post.array_size, post.neurons_num, dim_pre)

                    # can't specify a function with either side encoded connection
                    assert func == None 

                    case2 = connection.Case2(
                        (post.array_size, post.neurons_num,
                         pre.array_size, pre.neurons_num))
                    encoded_output = case2(transform, pre_output)

                    # pass in the pre population encoded output function
                    # to the post population, connecting them for theano
                    post.add_termination(name=pre_name, pstc=pstc, 
                        encoded_input=encoded_output)

                    return

        origin_socket, destination_socket = \
            zmq_utils.create_socket_defs_pushpull(pre.name, post.name)

        # if decoded-decoded connection (case 1)
        # compute transform if not given, if given make sure shape is correct
        transform = connection.compute_transform(
            dim_pre=dim_pre,
            dim_post=post.dimensions,
            array_size=post.array_size,
            weight=weight,
            index_pre=index_pre,
            index_post=index_post, 
            transform=transform)

        # pre output needs to be replaced during execution using IPC
        # pass pre_out and transform + calculate dot product in accumulator
        # passing VALUE of pre output (do not share theano shared vars between processes)
        post.add_termination(input_socket=destination_socket, name=pre_name, pstc=pstc, 
            decoded_input=pre_output.get_value(), transform=transform) 

        pre_origin.add_output(origin_socket)

    def get_object(self, name):
        """This is a method for parsing input to return the proper object.

        The only thing we need to check for here is a ':',
        indicating an origin.

        :param string name: the name of the desired object

        """
        assert isinstance(name, str)

        # separate into node and origin, if specified
        split = name.split(':')
        nodes = self.nodes

        if len(split) == 1:
            # no origin specified
            return nodes[name]
        elif len(split) == 2:
            # origin specified
            node = nodes[split[0]]
            return node.origin[split[1]]

    def get_origin(self, name, func=None):
        """This method takes in a string and returns the decoded_output function 
        of this object. If no origin is specified in name then 'X' is used.

        :param string name: the name of the object(and optionally :origin) from
                            which to take decoded_output from
        :returns: specified origin
        """
        obj = self.get_object(name) # get the object referred to by name

        if not isinstance(obj, origin.Origin):
            # if obj is not an origin, find the origin
            # the projection originates from

            # take default identity decoded output from obj population
            origin_name = 'X'

            if func is not None: 
                # if this connection should compute a function

                # set name as the function being calculated
                origin_name = func.__name__

                #TODO: better analysis to see if we need to build a new origin
                # (rather than just relying on the name)
                if origin_name not in obj.origin:
                    # if an origin for this function hasn't already been created
                    # create origin with to perform desired func
                    obj.add_origin(origin_name, func, dt=self.dt)

            obj = obj.origin[origin_name]

        else:
            # if obj is an origin, make sure a function wasn't given
            # can't specify a function for an already created origin
            assert func == None

        return obj

    def learn(self, pre, post, error, pstc=0.01, **kwargs):
        """Add a connection with learning between pre and post,
        modulated by error. Error can be a Node, or an origin. If no 
        origin is specified in the format node:origin, then 'X' is used.

        :param Ensemble pre: the pre-synaptic population
        :param Ensemble post: the post-synaptic population
        :param Ensemble error: the population that provides the error signal
        :param list weight_matrix:
            the initial connection weights with which to start

        """
        pre_name = pre
        pre = self.get_object(pre)
        post = self.get_object(post)
        error = self.get_origin(error)
        return post.add_learned_termination(name=pre_name, pre=pre, error=error, 
            pstc=pstc, **kwargs)

    def make(self, name, *args, **kwargs): 
        """Create and return an ensemble of neurons.

        Note that all ensembles are actually arrays of length 1.

        :param string name: name of the ensemble (must be unique)
        :param int seed:
            Random number seed to use.
            If this is None and the Network was constructed
            with a seed parameter, a seed will be randomly generated.
        :returns: the newly created ensemble      

        """
        if 'seed' not in kwargs.keys():
            if self.fixed_seed is not None:
                kwargs['seed'] = self.fixed_seed
            else:
                # if no seed provided, get one randomly from the rng
                kwargs['seed'] = self.random.randrange(0x7fffffff)

        ticker_socket, node_socket = \
            zmq_utils.create_socket_defs_reqrep("ticker", name)
        kwargs['dt'] = self.dt

        # create ensemble and ensemble process
        # TODO: currently using separate processes for direct nodes
        # (Terry wanted them on a single proceses, but need more info for that)
        e = ensemble.EnsembleProcess(name, node_socket, *args, **kwargs)
        p = Process(target=e.run, name=name)
        self.processes.append((p, ticker_socket.create_socket(self.zmq_context),))
        self.nodes[name] = e

        return e

    def make_array(self, name, neurons, array_size, dimensions=1, **kwargs):
        """Generate a network array specifically.

        This function is depricated; use for legacy code
        or non-theano API compatibility.
        """
        return self.make(
            name=name, neurons=neurons, dimensions=dimensions,
            array_size=array_size, **kwargs)

    def make_input(self, *args, **kwargs): 
        """Create an input and add it to the network."""
        i = input.Input(*args, **kwargs)
        self.add(i)
        return i

    def make_subnetwork(self, name):
        """Create a subnetwork.  This has no functional purpose other than
        to help organize the model.  Components within a subnetwork can
        be accessed through a dotted name convention, so an element B inside
        a subnetwork A can be referred to as A.B.       

        :param name: the name of the subnetwork to create        
        """
        return subnetwork.SubNetwork(name, self)

    def make_probe(self, target, name=None, dt_sample=0.01, data_type='decoded', **kwargs):
        """Add a probe to measure the given target.

        :param target: a Theano shared variable to record
        :param name: the name of the probe
        :param dt_sample: the sampling frequency of the probe
        :returns: The Probe object

        """

        raise Exception("ERROR: Probes are not yet supported by spike.")

        i = 0
        target_name = target + '-' + data_type
        while name is None or self.nodes.has_key(name):
            i += 1
            name = ("Probe%d" % i)

        # get the signal to record
        if data_type == 'decoded':
            target = self.get_origin(target).decoded_output

        elif data_type == 'spikes':
            target = self.get_object(target)
            # check to make sure target is an ensemble
            assert isinstance(target, ensemble.Ensemble)
            target = target.neurons.output
            # set the filter to zero
            kwargs['pstc'] = 0

        p = probe.Probe(name=name, target=target, target_name=target_name, 
            dt_sample=dt_sample, **kwargs)
        self.add(p)
        return p

    # TODO: remove this method if direct ensembles do not need to share processes
    def make_theano_tick(self):
        """Generate the theano function for running the network simulation.

        :returns: theano function
        """
        raise Exception("ERROR: Global theano tick not supported.")
        # dictionary for all variables
        # and the theano description of how to compute them 
        updates = OrderedDict()

        # for every direct ensemble in the network
        for node in self.direct_nodes.values():
            # if there is some variable to update
            if hasattr(node, 'update'):
                # add it to the list of variables to update every time step
                updates.update(node.update())

        # create graph and return optimized update function
        return theano.function([], [], updates=updates.items())

    def run(self, time):
        """Run the simulation.

        If called twice, the simulation will continue for *time*
        more seconds. Note that the ensembles are simulated at the
        dt timestep specified when they are created.

        :param float time: the amount of time (in seconds) to run
        :param float dt: the timestep of the update
        """
        if not self.setup:
            for p in self.processes:
                proc = p[0]
                if not proc.is_alive():
                    proc.start()
            self.setup = True

        for i in range(int(time / self.dt)):
            # get current time step
            t = self.run_time + i * self.dt

            ## Tick all nodes
            for p in self.processes:
                ticker_conn = p[1]
                ticker_conn.send(str(t))

            ## Wait for all nodes
            for j in self.processes:
                ticker_conn = j[1]
                ticker_conn.recv()

        for p in self.processes:
            ticker_conn = p[1]
            ticker_conn.send("END")

        # update run_time variable
        self.run_time += time

        self.clean_up()

    # called when the simulation is done (otherwise, procs will hang)
    def clean_up(self):
        # wait for all procs to end
        for p in self.processes:
            proc = p[0]
            proc.join()

    def write_data_to_hdf5(self, filename='data'):
        """This is a function to call after simulation that writes the 
        data of all probes to filename using the Neo HDF5 IO module.

        :param string filename: the name of the file to write out to
        """
        import neo
        from neo import hdf5io

        # get list of probes 
        probe_list = [self.nodes[node] for node in self.nodes 
                      if node[:5] == 'Probe']

        # if no probes then just return
        if len(probe_list) == 0: return

        # open up hdf5 file
        if not filename.endswith('.hd5'): filename += '.hd5'
        iom = hdf5io.NeoHdf5IO(filename=filename)

        #TODO: set up to write multiple trials/segments to same block 
        #      for trials run at different points
        # create the all encompassing block structure
        block = neo.Block()
        # create the segment, representing a trial
        segment = neo.Segment()
        # put the segment in the block
        block.segments.append(segment)

        # create the appropriate Neo structures from the Probes data
        #TODO: pair any analog signals and spike trains from the same
        #      population together into a RecordingChannel
        for probe in probe_list:
            # decoded signals become AnalogSignals
            if probe.target_name.endswith('decoded'):
                segment.analogsignals.append(
                    neo.AnalogSignal(
                        probe.get_data() * quantities.dimensionless, 
                        sampling_period=probe.dt_sample * quantities.s,
                        target_name=probe.target_name) )
            # spikes become spike trains
            elif probe.target_name.endswith('spikes'):
                # have to change spike train of 0s and 1s to list of times
                for neuron in probe.get_data().T:
                    segment.spiketrains.append(
                        neo.SpikeTrain(
                            [
                                t * probe.dt_sample 
                                for t, val in enumerate(neuron[0]) 
                                if val > 0
                            ] * quantities.s,
                            t_stop=len(probe.data),
                            target_name=probe.target_name) )
            else: 
                print 'Do not know how to write %s to NeoHDF5 file'%probe.target_name
                assert False

        # write block to file
        iom.save(block)
        # close up hdf5 file
        iom.close()
