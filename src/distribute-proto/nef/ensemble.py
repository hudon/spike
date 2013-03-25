from theano.tensor.shared_randomstreams import RandomStreams
from theano import tensor as TT
import theano
import numpy

import neuron
import origin

#  TODO:  Split Up Ensembles
#  One method for splitting up ensembles would be to come up with a second concept of
#  EnsemblePartitions.  The Ensemble class would retain the same exernal interface and 
#  be responsible for partitioning the required Ensemble into EnsemblePartitions, when
#  the Ensemble class is initialized, and consolidating the output from each EnseblePartition
#  into the the output required by the external Enseble interface.
#  An EnsemblePartiton would essentially be what an ensemble is now.  The ensemble class would
#  be responsible for deciding how to represent the larger enseble in terms of a series of
#  smaller ones.
#  Implementing this requires understanding the intimate details of the mathematical operations 
#  that calculate the output.  Clearly, many of the calculations involve matrix operations, so 
#  it is not as simple as dividing all arrays in half.  Dimension and cardinality of involved
#  data structures must be known exactly.  I have attempted to document these specifics where 
#  I can.

# generates a set of encoders
def make_encoders(neurons,dimensions,srng,encoders=None):
    if encoders is None:
	#  This is that same theano RandomStream thing that is standing in for
	#  http://docs.scipy.org/doc/numpy/reference/generated/numpy.random.RandomState.normal.html#numpy.random.RandomState.normal
	#  size : tuple of ints Output shape. If the given shape is, e.g., (m, n, k), then m * n * k samples are drawn.
	#  SUMMARY:  Returns a random matrix with values from a normal distribution.  Size is R X C = dimensions X neurons
        encoders=srng.normal((neurons,dimensions))
    else:
        encoders=numpy.array(encoders)
	#  numpy.tile:  Construct an array by repeating A the number of times given by reps.
	#  It producs a matrix of Size is R X C = dimensions X neurons
        encoders=numpy.tile(encoders,(neurons/len(encoders)+1,1))[:neurons,:dimensions]

    #  I don't understand how you can square the encoders here, because the dimensions don't match up
    #  Thus, I don't the shape of the result
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
            type = 'lif', dt = 0.001, encoders = None, name = None):
        self.seed = seed
        self.neurons = neurons
        self.dimensions = dimensions
        self.count = count
        self.name = name

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

        # compute alpha and bias
        srng = RandomStreams(seed=seed)
	#  uniform(self, size=(), low=0.0, high=1.0, ndim=None):
	#  Sample a tensor of given size whose element from a uniform distribution between low and high.
	#  FROM http://deeplearning.net/software/theano/library/tensor/raw_random.html#raw_random.RandomStreamsBase
	#  This is a symbolic stand-in for numpy.random.RandomState.
	#  http://docs.scipy.org/doc/numpy/reference/generated/numpy.random.RandomState.uniform.html#numpy.random.RandomState.uniform
	#  size : int or tuple of ints, optional Shape of output. If the
	#  given size is, for example, (m,n,k), m*n*k samples are generated.
	#  If no shape is specified, a single sample is returned.
	#  SUMMARY:  srng.uniform generates a random sample array of length [neurons] (I think)
        max_rates = srng.uniform([neurons], low=max_rate[0], high=max_rate[1])
        threshold = srng.uniform([neurons], low=intercept[0], high=intercept[1])
	#  I think this is returning alpha and bias as an array of length [neurons]
        alpha, self.bias = theano.function([], self.neuron.make_alpha_bias(max_rates,threshold))()
        self.bias = self.bias.astype('float32')

        # compute encoders
	#  The actual dimensions of encoders is not obvious to me
        self.encoders = make_encoders(neurons, dimensions, srng, encoders=encoders)
        self.encoders = (self.encoders.T * alpha).T

        # make default origin
	#  Again, origin uses insane math and I can't figure out shape it has
	#  so I don't really know how/if to split it up.
        self.origin = dict(X=origin.Origin(self))
        self.accumulator = {}

    # create a new origin that computes a given function
    def add_origin(self, name, func):
        self.origin[name] = origin.Origin(self, func)

    # create a new termination that takes the given input (a theano object)
    # and filters it with the given tau
    def add_input(self, input_pipe, tau, value_size, transform):
        if tau not in self.accumulator:
            self.accumulator[tau] = Accumulator(self, tau)

        self.accumulator[tau].add(input_pipe, value_size, transform)

    def make_tick(self):
	#  Create a dictionary called updates
        updates = {}
	#  Call the update method of dictionary, pass in a crazy tensor
        updates.update(self.update())
	#  Call our theano function interface, passing in the dictionary updates
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
	#  I have no idea how to split this up while keeping it mathematically correct
        if len(self.accumulator) > 0:
            X = sum(a.new_value for a in self.accumulator.values())
            #  reshape gives a new shape to an array without changing its data.
            #  http://docs.scipy.org/doc/numpy/reference/generated/numpy.reshape.html
            X = X.reshape((self.count, self.dimensions))
            #  self.encoders.T is the transpose of self.encoders
            #  http://docs.scipy.org/doc/numpy-1.5.x/reference/generated/numpy.ndarray.T.html#numpy.ndarray.T
            #  TT.dot calculates the inner tensor product of X and self.encoders.T
            #  http://deeplearning.net/software/theano/library/tensor/basic.html#tensor.dot
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



