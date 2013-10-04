import numpy as np
from _collections import OrderedDict

from . import neuron
from . import ensemble_origin
from . import origin
from . import filter
from .hPES_termination import hPESTermination

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

        print "EnsembleProcess tick function.  Finished query poll state.",self.name

        inputs = {}
        for i, socket in enumerate(self.input_sockets):
            print "EnsembleProcess tick function.  -----",self.n,"----- Before recv_pyobj.",self.name
            val = socket.get_instance().recv_pyobj()
            print "EnsembleProcess tick function.  -----",self.n,"----- After recv_pyobj.",self.name
            inputs[socket.name] = val

        self.n = self.n + 1
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

            # make default origin
            self.add_origin('X', func=None, dt=dt, eval_points=self.eval_points) 

    def add_termination(self, name, pstc, decoded_input=None, 
        encoded_input=None, input_socket=None, transform=None, case=None):
        pass

    def add_origin(self, name, func, **kwargs):
        if self.mode == 'spiking':
            if 'eval_points' not in kwargs.keys():
                kwargs['eval_points'] = self.eval_points
            self.origin[name] = ensemble_origin.EnsembleOrigin(
                ensemble=self, func=func, **kwargs)

    def get_unique_name(self, name, dic):
        i = 0
        while dic.has_key(name + '_' + str(i)): 
            i += 1

        return name + '_' + str(i)

    def make_tick(self):
        # introduce 1-time-tick delay
        for o in self.origin.values():
            if o.func is not None and self.mode == 'direct': continue
            o.tick()

    # Using the dt that was passed to the ensemble at construction time
    def update(self):
        return OrderedDict()
