from theano.tensor.shared_randomstreams import RandomStreams
from theano import tensor as TT
import theano
import numpy

import neuron
import origin

# generates a set of encoders
def make_encoders(neurons,dimensions,srng,encoders=None):
    if encoders is None:
        encoders = srng.normal((neurons, dimensions))
    else:
        encoders = numpy.array(encoders)
	#  numpy.tile:  Construct an array by repeating A the number of times given by reps.
	#  It producs a matrix of Size is R X C = dimensions X neurons
        encoders = numpy.tile(encoders, (neurons / len(encoders) + 1, 1))[:neurons, :dimensions]

    norm = TT.sum(encoders * encoders, axis=[1], keepdims=True)
    encoders = encoders / TT.sqrt(norm)
    return theano.function([], encoders)()

# a collection of terminations, all sharing the same time constant
class Accumulator:
    def __init__(self, ensemble, tau):
        self.ensemble = ensemble   # the ensemble this set of terminations is attached to

        self.value = theano.shared(numpy.zeros(self.ensemble.dimensions * self.ensemble.count).astype('float32'))  # the current filtered value

        self.decay = numpy.exp(-self.ensemble.neuron.dt / tau)   # time constant for filter
        self.total = None   # the theano object representing the sum of the inputs to this filter

        # parallel lists
        self.input_pipes = []
        self.vals = []

    def add(self, input_pipe, value_size, transform=None):
        self.input_pipes.append(input_pipe)

        val = theano.shared(numpy.zeros(value_size).astype('float32'))
        self.vals.append(val)

        if transform is not None:
            val = TT.dot(val, transform)

        if self.total is None:
            self.total = val
        else:
            self.total = self.total + val

        self.new_value = self.decay * self.value + (1 - self.decay) * self.total

    # returns False if some data was not available
    def tick(self):
        for pipe in self.input_pipes:
            if not pipe.poll():
                return False

        for i, pipe in enumerate(self.input_pipes):
            val = pipe.recv()
            self.vals[i].set_value(val)
     
        return True

class Ensemble:
    def __init__(self, neurons, dimensions, count = 1, max_rate = (200, 300),
            intercept = (-1.0, 1.0), t_ref = 0.002, t_rc = 0.02, seed = None,
            type='lif', dt=0.001, encoders=None,
            is_subensemble=False, name=None, decoders=None, bias=None):
        self.name = name
        self.seed = seed

        self.neurons = neurons
        self.dimensions = dimensions
        self.count = count
        self.accumulator = {}

        self.is_subensemble = is_subensemble

        # create the neurons
        # TODO: handle different neuron types, which may have different parameters to pass in
    	#  The structure of the data contained in self.neuron consists of several variables that are 
    	#  arrays of the form
    	#  Array([
    	#	[x_0_0, x_0_1, x_0_2,..., x_0_(neurons - 1)],
    	#	[x_1_0, x_1_1, x_1_2,..., x_1_(neurons - 1)],
    	#	[...],
    	#	[x_(count-1)_0, x_(count-1)_1, x_(count-1)_2,..., x_$count-1)_(neurons - 1)]
    	#  ])
        self.neuron = neuron.names[type]((count, self.neurons), t_rc = t_rc, t_ref = t_ref, dt = dt)

        if is_subensemble:
            self.bias = bias
            self.encoders = encoders
        else:
            # compute alpha and bias
            srng = RandomStreams(seed=seed)
            max_rates = srng.uniform([neurons], low=max_rate[0], high=max_rate[1])
            threshold = srng.uniform([neurons], low=intercept[0], high=intercept[1])

            alpha, self.bias = theano.function([], self.neuron.make_alpha_bias(max_rates, threshold))()
            self.bias = self.bias.astype('float32')

            # compute encoders
            self.encoders = make_encoders(neurons, dimensions, srng, encoders=encoders)
            self.encoders = (self.encoders.T * alpha).T

        # make default origin
        self.origin = dict(X=origin.Origin(self, decoder=decoders))

    def get_subensemble_parts(self, num_parts):
        """
        Uses an encoder, decoder and bias of an ensemble and divides them
        into the specified number of parts for the subensembles.
        """
        parts = []

        decoder_parts = self.get_subensemble_decoder(num_parts, "X")

        encoder_length = len(self.encoders)
        bias_length = len(self.bias)

        # create the specified number of subensembles
        for e_num in range(1, num_parts + 1):
            e_size = encoder_length / num_parts
            b_size = bias_length / num_parts

            encoder_part = self.encoders[e_size * (e_num - 1):e_size * e_num]
            bias_part = self.bias[b_size * (e_num - 1):b_size * e_num]

            parts.append((encoder_part, decoder_parts[e_num - 1], bias_part))

        return parts

    def get_subensemble_decoder(self, num_parts, origin_name, func=None):
        """ Gets decoder for a specified origin of the ensemble
        and divides it into the specified number of parts for subensembles
        """
        parts = []

        # TODO do not require an Origin to be created just to compute decoder
        if origin_name not in self.origin:
            # create the origin in order to compute a decoder
            self.add_origin(origin_name, func)
            # print "name " + self.name + " decoder: " + str(self.origin[origin_name].decoder)

        decoder = self.origin[origin_name].decoder
        decoder_length = len(decoder)

        # create the specified number of decoders
        for e_num in range(1, num_parts + 1):
            d_size = decoder_length / num_parts
            decoder_part = decoder[d_size * (e_num - 1):d_size * e_num]

            parts.append(decoder_part)

        return parts


    # create a new origin that computes a given function
    def add_origin(self, name, func, decoder=None):
        self.origin[name] = origin.Origin(self, func, decoder=decoder)

    # create a new termination that takes the given input (a theano object)
    # and filters it with the given tau
    def add_input(self, input_pipe, tau, value_size, transform):
        if tau not in self.accumulator:
            self.accumulator[tau] = Accumulator(self, tau)

        self.accumulator[tau].add(input_pipe, value_size, transform)

    def make_tick(self):
        updates = {}
        updates.update(self.update())
        self.theano_tick = theano.function([], [], updates = updates)

    def tick(self):
        # start the tick in the accumulators
        for a in self.accumulator.values():
            if not a.tick():
                # no data was in the pipe
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
            #  reshape gives a new shape to an array without changing its data.
            X = X.reshape((self.count, self.dimensions))
            #  self.encoders.T is the transpose of self.encoders
            #  TT.dot calculates the inner tensor product of X and self.encoders.T
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

    def run(self, ticker_conn):
	#  While true because once we have run the model for enough time, we
	#  call a method from the user application to kill these processes
        while True:
            ticker_conn.recv()
            self.tick()
            ticker_conn.send(1)



