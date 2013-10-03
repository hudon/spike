import theano
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano import tensor as TT
import numpy as np
from _collections import OrderedDict

from . import neuron
from . import ensemble_origin
from . import origin
from . import cache
from . import filter
from .hPES_termination import hPESTermination
from .helpers import map_gemv

from multiprocessing import Process

from pprint import pprint

import os
import zmq
import zmq_utils

class EnsembleProcess(Process):
    """ A NEFProcess is a wrapper for an ensemble or sub-ensemble. It is
    responsible for infrastructure logic such as setting up messaging,
    printing, process clean-up, etc. It also acts as an Adapter for most of
    the Ensemble's methods.

    :param str name: name of the process
    """
    def __init__(self, name, ticker_socket_def, *args, **kwargs):
        super(EnsembleProcess, self).__init__(target=self.run, name=name)
        self.name = name

        ## Adapter for Ensemble
        self.ensemble = Ensemble(*args, **kwargs)

        self.origin = self.ensemble.origin
        self.dimensions = self.ensemble.dimensions
        self.array_size = self.ensemble.array_size
        self.neurons_num = self.ensemble.neurons_num
        self.add_origin = self.ensemble.add_origin
        self.update = self.ensemble.update

        # context should be created when the process is started (bind_sockets)
        self.zmq_context = None
        self.poller = zmq.Poller()

        self.unique_socket_names = {}
        self.input_sockets = []
        self.ticker_socket_def = ticker_socket_def
        self.n = 0

    def bind_sockets(self):
        # create a context for this ensemble process if do not have one already
        if self.zmq_context is None:
            self.zmq_context = zmq.Context()

        # create socket connections for inputs
        for socket in self.input_sockets:
            socket.init(self.zmq_context)
            self.poller.register(socket.get_instance(), zmq.POLLIN)

        for o in self.ensemble.origin.values():
            o.bind_sockets(self.zmq_context)

        self.ticker_conn = self.ticker_socket_def.create_socket(self.zmq_context)

    def tick(self):
        """ This process tick is responsible for IPC, keeping the Ensemble
        unaware of the details of messaging/sockets.
        """

        # poll for all inputs, do not continue unless all inputs are available
        print "EnsembleProcess tick function.  About to query poll state.",self.name
        is_waiting_for_input = True
        ggg = 0
        while is_waiting_for_input:
            ggg = ggg + 1
            is_waiting_for_input = False
            responses = dict(self.poller.poll(1))
            for i, socket in enumerate(self.input_sockets):
                socket_inst = socket.get_instance()
                if socket_inst not in responses or responses[socket_inst] != zmq.POLLIN:
                    is_waiting_for_input = True
            if (ggg > 3000):
                print "Over 3000 times, hang forever.",os.getpid()," ",self.name
                for i, socket in enumerate(self.input_sockets):
                    socket_inst = socket.get_instance()
                    if socket_inst not in responses:
                        print socket.name," not in responses.",os.getpid()," ",self.name
                    if socket_inst in responses and responses[socket_inst] != zmq.POLLIN:
                        print socket.name," was zmq.POLLIN.",os.getpid()," ",self.name
                while True:
                   pass


        inputs = {}
        for i, socket in enumerate(self.input_sockets):
            print "EnsembleProcess tick function.  -----",self.n,"----- Before recv_pyobj.",self.name
            val = socket.get_instance().recv_pyobj()
            print "EnsembleProcess tick function.  -----",self.n,"----- After recv_pyobj.",self.name
            inputs[socket.name] = val

        self.n = self.n + 1
        self.ensemble.tick(inputs)
        print "EnsembleProcess tick function.  After self.ensemble.tick.",os.getpid()," ",self.name

    def run(self):
        print "EnsembleProcess run Before bind_sockets.",os.getpid()," ",self.name
        self.bind_sockets()
        print "EnsembleProcess run After bind_sockets.",os.getpid()," ",self.name
        self.ensemble.make_tick()

        print "EnsembleProcess run function getting time.  Before recv.",os.getpid()," ",self.name
        time = float(self.ticker_conn.recv())
        print "EnsembleProcess run function getting time.  After recv.",os.getpid()," ",self.name

        for i in range(int(time / self.ensemble.dt)):
            #print "EnsembleProcess run Before self.tick with i=",i," ",os.getpid()," ",self.name
            self.tick()
            #print "EnsembleProcess run After self.tick with i=",i," ",os.getpid()," ",self.name

        print "EnsembleProcess run function sending FIN.  Before send.",os.getpid()," ",self.name
        self.ticker_conn.send("FIN") # inform main proc that ens finished
        print "EnsembleProcess run function sending FIN.  After send.",os.getpid()," ",self.name
        print "EnsembleProcess run function getting ACK.  Before recv.",os.getpid()," ",self.name
        self.ticker_conn.recv() # wait for an ACK from main proc before exiting
        print "EnsembleProcess run function getting ACK.  After recv.",os.getpid()," ",self.name

    def add_termination(self, input_socket, *args, **kwargs):
        ## We get a unique name for the inputs so that the ensemble doesn't
        ## need to do so and we can then use the unique_name to identify
        ## the input sockets
        unique_name = self.ensemble.get_unique_name(kwargs['name'],
                self.unique_socket_names)

        self.unique_socket_names[unique_name] = ""
        kwargs['name'] = unique_name

        self.input_sockets.append(zmq_utils.Socket(input_socket, unique_name))

        return self.ensemble.add_termination(*args, **kwargs)


class Ensemble:
    """An ensemble is a collection of neurons representing a vector space.

    """

    def __init__(self, neurons, dimensions, dt, tau_ref=0.002, tau_rc=0.02,
                 max_rate=(200, 300), intercept=(-1.0, 1.0), radius=1.0,
                 encoders=None, seed=None, neuron_type='lif',
                 array_size=1, eval_points=None, decoder_noise=0.1,
                 noise_type='uniform', noise=None, mode='spiking'):
        """Construct an ensemble composed of the specific neuron model,
        with the specified neural parameters.

        :param int neurons: number of neurons in this population
        :param int dimensions:
            number of dimensions in the vector space
            that these neurons represent
        :param float tau_ref: length of refractory period
        :param float tau_rc:
            RC constant; approximately how long until 2/3
            of the threshold voltage is accumulated
        :param tuple max_rate:
            lower and upper bounds on randomly generated
            firing rates for each neuron
        :param tuple intercept:
            lower and upper bounds on randomly generated
            x offsets for each neuron
        :param float radius:
            the range of input values (-radius:radius)
            per dimension this population is sensitive to
        :param list encoders: set of possible preferred directions
        :param int seed: seed value for random number generator
        :param string neuron_type:
            type of neuron model to use, options = {'lif'}
        :param int array_size: number of sub-populations for network arrays
        :param list eval_points:
            specific set of points to optimize decoders over by default
        :param float decoder_noise: amount of noise to assume when computing 
            decoder    
        :param string noise_type:
            the type of noise added to the input current.
            Possible options = {'uniform', 'gaussian'}.
            Default is 'uniform' to match the Nengo implementation.
        :param float noise:
            noise parameter for noise added to input current,
            sampled at every timestep.
            If noise_type = uniform, this is the lower and upper
            bound on the distribution.
            If noise_type = gaussian, this is the variance.

        """
        self.dt = dt

        if seed is None:
            seed = np.random.randint(1000)
        self.seed = seed
        self.neurons_num = neurons
        self.dimensions = dimensions
        self.array_size = array_size
        self.radius = radius
        self.noise = noise
        self.noise_type = noise_type
        self.decoder_noise = decoder_noise

        self.mode = mode

        # make sure that eval_points is the right shape
        if eval_points is not None:
            eval_points = np.array(eval_points)
            if len(eval_points.shape) == 1:
                eval_points.shape = [1, eval_points.shape[0]]
        self.eval_points = eval_points

        # make sure intercept is the right shape
        if isinstance(intercept, (int,float)): intercept = [intercept, 1]
        elif len(intercept) == 1: intercept.append(1) 

        self.cache_key = cache.generate_ensemble_key(neurons=neurons, 
            dimensions=dimensions, tau_rc=tau_rc, tau_ref=tau_ref, 
            max_rate=max_rate, intercept=intercept, radius=radius, 
            encoders=encoders, decoder_noise=decoder_noise, 
            eval_points=eval_points, noise=noise, seed=seed, dt=dt)

        # make dictionary for origins
        self.origin = {}
        # set up a dictionary for decoded_input
        self.decoded_input = {}

        # if we're creating a spiking ensemble
        if self.mode == 'spiking': 

            # TODO: handle different neuron types,
            self.neurons = neuron.types[neuron_type](
                size=(array_size, self.neurons_num),
                tau_rc=tau_rc, tau_ref=tau_ref)

            # compute alpha and bias
            self.srng = RandomStreams(seed=seed)
            self.max_rate = max_rate
            max_rates = self.srng.uniform(
                size=(self.array_size, self.neurons_num),
                low=max_rate[0], high=max_rate[1])  
            threshold = self.srng.uniform(
                size=(self.array_size, self.neurons_num),
                low=intercept[0], high=intercept[1])


            # set up a dictionary for encoded_input connections
            self.encoded_input = {}
            # list of learned terminations on ensemble
            self.learned_terminations = []

            # make default origin
            self.add_origin('X', func=None, dt=dt, eval_points=self.eval_points) 

        elif self.mode == 'direct':
            # make default origin
            self.add_origin('X', func=None, dimensions=self.dimensions*self.array_size) 
            # reset neurons_num to 0
            self.neurons_num = 0

    def add_termination(self, name, pstc, decoded_input=None, 
        encoded_input=None, input_socket=None, transform=None, case=None):
        pass

    def add_learned_termination(self, name, pre, error, pstc, 
                                learned_termination_class=hPESTermination,
                                **kwargs):
        """Adds a learned termination to the ensemble.

        Input added to encoded_input, and a learned_termination object
        is created to keep track of the pre and post
        (self) spike times, and adjust the weight matrix according
        to the specified learning rule.

        :param Ensemble pre: the pre-synaptic population
        :param Ensemble error: the Origin that provides the error signal
        :param float pstc:
        :param learned_termination_class:
        """
        raise Exception("ERRPR", "Learned connections are not usable yet.")

        #TODO: is there ever a case we wouldn't want this?
        assert error.dimensions == self.dimensions * self.array_size

        # generate an initial weight matrix if none provided,
        # random numbers between -.001 and .001
        if 'weight_matrix' not in kwargs.keys():
            weight_matrix = np.random.uniform(
                size=(self.array_size * pre.array_size,
                      self.neurons_num, pre.neurons_num),
                low=-.001, high=.001)
            kwargs['weight_matrix'] = weight_matrix
        else:
            # make sure it's an np.array
            #TODO: error checking to make sure it's the right size
            kwargs['weight_matrix'] = np.array(kwargs['weight_matrix']) 

        learned_term = learned_termination_class(
            pre=pre, post=self, error=error, **kwargs)

        learn_projections = [TT.dot(
            pre.neurons.output[learned_term.pre_index(i)],  
            learned_term.weight_matrix[i % self.array_size]) 
            for i in range(self.array_size * pre.array_size)]

        # now want to sum all the output to each of the post ensembles 
        # going to reshape and sum along the 0 axis
        learn_output = TT.sum( 
            TT.reshape(learn_projections, 
            (pre.array_size, self.array_size, self.neurons_num)), axis=0)
        # reshape to make it (array_size x neurons_num)
        learn_output = TT.reshape(learn_output, 
            (self.array_size, self.neurons_num))

        # the input_current from this connection during simulation
        self.add_termination(name=name, pstc=pstc, encoded_input=learn_output)
        self.learned_terminations.append(learned_term)
        return learned_term

    def add_origin(self, name, func, **kwargs):
        """Create a new origin to perform a given function
        on the represented signal.

        :param string name: name of origin
        :param function func:
            desired transformation to perform over represented signal
        :param list eval_points:
            specific set of points to optimize decoders over for this origin
        """

        # Create an ensemble_origin with decoders
        if self.mode == 'spiking':
            if 'eval_points' not in kwargs.keys():
                kwargs['eval_points'] = self.eval_points
            self.origin[name] = ensemble_origin.EnsembleOrigin(
                ensemble=self, func=func, **kwargs)

        # if we're in direct mode then this population is just directly 
        # performing the specified function, use a basic origin
        elif self.mode == 'direct':
            if func is not None:
                if 'initial_value' not in kwargs.keys():
                    # [func(np.zeros(self.dimensions)) for i in range(self.array_size)]
                    init = func(np.zeros(self.dimensions))
                    init = np.array([init for i in range(self.array_size)])
                    kwargs['initial_value'] = init.flatten()

            if kwargs.has_key('dt'): del kwargs['dt']
            self.origin[name] = origin.Origin(func=func, **kwargs) 

    def get_unique_name(self, name, dic):
        i = 0
        while dic.has_key(name + '_' + str(i)): 
            i += 1

        return name + '_' + str(i)

    def make_tick(self):
        updates = OrderedDict()
        updates.update(self.update())

        # introduce 1-time-tick delay
        for o in self.origin.values():
            if o.func is not None and self.mode == 'direct': continue
            o.tick()

    def direct_mode_tick(self):
        if self.mode == 'direct':
            # set up matrix to store accumulated decoded input
            X = np.zeros((self.array_size, self.dimensions))

            for di in self.decoded_input.values(): 
                # add its values to the total decoded input
                X += di.value.get_value()
            
            # if we're calculating a function on the decoded input
            for o in self.origin.values():
                if o.func is not None:
                    val = np.float32([o.func(X[i]) for i in range(len(X))])
                    o.decoded_output.set_value(val.flatten())
        else:
            raise Exception("ERROR", "The current ensemble does not have 'direct' mode.")

    # Receive the outputs of pre - decoded output - and pass it to filters
    def tick(self, inputs):
        # continue the tick in the origins
        for o in self.origin.values():
            print "Ensemble tick function.  Before o.tick.",os.getpid()
            o.tick()
            print "Ensemble tick function.  After o.tick.",os.getpid()
        print "Ensemble tick function.  Completed o.ticks.",os.getpid()

    # Using the dt that was passed to the ensemble at construction time
    def update(self):
        X = None 
        updates = OrderedDict()

        for ii, di in enumerate(self.decoded_input.values()):
            # add its values to the total decoded input
            if ii == 0:
                X = di.value
            else:
                X += di.value

            updates.update(di.update(self.dt))

        # if we're in spiking mode, then look at the input current and 
        # calculate new neuron activities for output
        if self.mode == 'spiking':

            for l in self.learned_terminations:
                # also update the weight matrices on learned terminations
                updates.update(l.update(self.dt))

        if self.mode == 'direct': 
            # if we're in direct mode then just directly pass the decoded_input 
            # to the origins for decoded_output
            for o in self.origin.values(): 
                if o.func is None:
                    if len(self.decoded_input) > 0:
                        updates.update(OrderedDict({o.decoded_output: 
                            TT.flatten(X).astype('float32')}))
        return updates
