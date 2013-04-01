from theano.tensor.shared_randomstreams import RandomStreams
from theano import tensor as TT
import theano
import numpy

import neuron
import origin
import zmq
import zmq_utils

# generates a set of encoders
def make_encoders(neurons,dimensions,srng,encoders=None):
    if encoders is None:
        encoders=srng.normal((neurons,dimensions))
    else:
        encoders=numpy.array(encoders)
        encoders=numpy.tile(encoders,(neurons/len(encoders)+1,1))[:neurons,:dimensions]

    norm=TT.sum(encoders*encoders,axis=[1],keepdims=True)
    encoders=encoders/TT.sqrt(norm)
    return theano.function([],encoders)()


# a collection of terminations, all sharing the same time constant
class Accumulator:
    def __init__(self, ensemble, tau):
        self.ensemble = ensemble   # the ensemble this set of terminations is attached to

        self.value = theano.shared(numpy.zeros(self.ensemble.dimensions * self.ensemble.count).astype('float32'))  # the current filtered value

        self.decay = numpy.exp(-self.ensemble.neuron.dt / tau)   # time constant for filter
        self.total = None   # the theano object representing the sum of the inputs to this filter

        # parallel lists
        self.input_socket_definitions = []
        self.input_sockets = []
        self.vals = []

    def add(self, input_socket_definition, value_size, transform=None):
        self.input_socket_definitions.append(input_socket_definition)

        val = theano.shared(numpy.zeros(value_size).astype('float32'))
        self.vals.append(val)

        if transform is not None:
            val = TT.dot(val, transform)

        if self.total is None:
            self.total = val
        else:
            self.total = self.total + val

        self.new_value = self.decay * self.value + (1 - self.decay) * self.total

    # Must be run prior to calling tick() to create and bind sockets
    def bind_sockets(self):
        for defn in self.input_socket_definitions:
            self.input_sockets.append(defn.create_socket())

    # returns False if some data was not available
    def tick(self):
        poller = zmq.Poller()

        for socket in self.input_sockets:
            poller.register(socket, zmq.POLLIN)

        responses = dict(poller.poll(1000))

        for i, socket in enumerate(self.input_sockets):
            if socket in responses and responses[socket] == zmq.POLLIN:
                print("YESPOLL")
                val = socket.recv()
                self.vals[i].set_value(val)
            else:
                print("NOPOLL")
                return False

        return True

class Ensemble:
    def __init__(self, neurons, dimensions, count = 1, max_rate = (200, 300),
            intercept = (-1.0, 1.0), t_ref = 0.002, t_rc = 0.02, seed = None,
            type = 'lif', dt = 0.001, encoders = None, name = None, address = "localhost"):
        self.seed = seed
        self.neurons = neurons
        self.dimensions = dimensions
        self.count = count
        self.name = name
        self.address = address
        self.ticker_conn = None

        # create the neurons
        # TODO: handle different neuron types, which may have different parameters to pass in
        self.neuron = neuron.names[type]((count, self.neurons), t_rc = t_rc, t_ref = t_ref, dt = dt)

        # compute alpha and bias
        srng = RandomStreams(seed=seed)
        max_rates = srng.uniform([neurons], low=max_rate[0], high=max_rate[1])
        threshold = srng.uniform([neurons], low=intercept[0], high=intercept[1])
        alpha, self.bias = theano.function([], self.neuron.make_alpha_bias(max_rates,threshold))()
        self.bias = self.bias.astype('float32')

        # compute encoders
        self.encoders = make_encoders(neurons, dimensions, srng, encoders=encoders)
        self.encoders = (self.encoders.T * alpha).T

        # make default origin
        self.origin = dict(X=origin.Origin(self))
        self.accumulator = {}

    # create a new origin that computes a given function
    def add_origin(self, name, func):
        self.origin[name] = origin.Origin(self, func)

    # create a new termination that takes the given input (a theano object)
    # and filters it with the given tau
    def add_input(self, input_socket_definition, tau, value_size, transform):
        if tau not in self.accumulator:
            self.accumulator[tau] = Accumulator(self, tau)

        self.accumulator[tau].add(input_socket_definition, value_size, transform)

    def make_tick(self):
        updates = {}
        updates.update(self.update())
        self.theano_tick = theano.function([], [], updates = updates)

    def tick(self):
        # start the tick in the accumulators
        for a in self.accumulator.values():
            if not a.tick():
                # no data was in the socket
                return
        self.theano_tick()
        # continue the tick in the origins
        for o in self.origin.values():
            o.tick()

    # compute the set of theano updates needed for this ensemble
    def update(self):
        # apply the bias to all neurons in the array
        input = numpy.tile(self.bias, (self.count, 1))

        # increase the input by the total of all the accumulators times the encoders
        # accumulator: group of terminations (inputs) that share the same low
        # pass filter. The group produces one result to feed into the
        # ensemble.
        if len(self.accumulator) > 0:
            X = sum(a.new_value for a in self.accumulator.values())
            X = X.reshape((self.count, self.dimensions))
            input = input + TT.dot(X, self.encoders.T)

        # pass that total into the neuron model to produce the main theano computation
        updates = self.neuron.update(input)

        # also update the filter values in the accumulators
        for a in self.accumulator.values():
            updates[a.value] = a.new_value.astype('float32')

        # and compute the decoded origin values from the neuron output
        for o in self.origin.values():
            updates.update(o.update(updates[self.neuron.output]))

        return updates

    def bind_sockets(self):
        # Create socket connections for inputs
        for a in self.accumulator.values():
            a.bind_sockets()

        for o in self.origin.values():
            o.bind_sockets()

        # zmq.REP strictly enforces alternating recv/send ordering
        zmq_context = zmq.Context()
        self.ticker_conn = zmq_context.socket(zmq.REP)
        self.ticker_conn.connect(zmq_utils.TICKER_SOCKET_LOCAL_NAME)


    def run(self):
        self.bind_sockets()

        while True:
            self.ticker_conn.recv()
            self.tick()
            self.ticker_conn.send("")
