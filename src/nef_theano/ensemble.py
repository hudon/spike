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

import zmq
import zmq_utils

class EnsembleProcess(object):
    """ A NEFProcess is a wrapper for an ensemble or sub-ensemble. It is
    responsible for infrastructure logic such as setting up messaging,
    printing, process clean-up, etc. It also acts as an Adapter for most of
    the Ensemble's methods.

    :param str name: name of the process
    """
    def __init__(self, name, *args, **kwargs):
        self.name = name

        ## Adapter for Ensemble
        self.ensemble = Ensemble(*args, **kwargs)

        self.origin = self.ensemble.origin
        self.dimensions = self.ensemble.dimensions
        self.array_size = self.ensemble.array_size
        self.neurons_num = self.ensemble.neurons_num

        # context should be created when the process is started (bind_sockets)
        self.zmq_context = None
        self.poller = zmq.Poller()

        self.unique_socket_names = {}
        self.input_sockets = []

    def add_origin(self, name, func, **kwargs):
        self.ensemble.add_origin(name, func, **kwargs)

    def update(self):
        return self.ensemble.update()

    def bind_sockets(self, admin_socket_def):
        # create a context for this ensemble process if do not have one already
        if self.zmq_context is None:
            self.zmq_context = zmq.Context()

        # create socket connections for inputs
        for socket in self.input_sockets:
            socket.init(self.zmq_context)
            self.poller.register(socket.get_instance(), zmq.POLLIN)

        for o in self.ensemble.origin.values():
            o.bind_sockets(self.zmq_context)

        self.admin_conn = admin_socket_def.create_socket(self.zmq_context)

    def tick(self):
        """ This process tick is responsible for IPC, keeping the Ensemble
        unaware of the details of messaging/sockets.
        """

        # poll for all inputs, do not continue unless all inputs are available
        is_waiting_for_input = True
        while is_waiting_for_input:
            is_waiting_for_input = False
            responses = dict(self.poller.poll(1))
            for i, socket in enumerate(self.input_sockets):
                socket_inst = socket.get_instance()
                if socket_inst not in responses or responses[socket_inst] != zmq.POLLIN:
                    is_waiting_for_input = True

        inputs = {}
        for i, socket in enumerate(self.input_sockets):
            val = socket.get_instance().recv_pyobj()
            inputs[socket.name] = val

        self.ensemble.tick(inputs)

    def run(self, admin_socket_def, time):
        self.bind_sockets(admin_socket_def)
        self.ensemble.make_tick()

        for i in range(int(time / self.ensemble.dt)):
            self.tick()

        self.admin_conn.recv_pyobj() # FIN
        self.admin_conn.send_pyobj({'result': 'ack'})

        self.admin_conn.recv_pyobj() # KILL

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
        noise_type='uniform', noise=None, mode='spiking',
        is_subensemble=False, decoders=None, bias=None, alpha=None):
        self.dt = dt

        np.random.seed(97)
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
        self.is_subensemble = is_subensemble

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
            encoders=(encoders if not is_subensemble else None),
            decoder_noise=decoder_noise,
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

            if is_subensemble:
                self.bias = bias
                self.alpha = alpha
                self.encoders = encoders
            else:
                max_rates = self.srng.uniform(
                    size=(self.array_size, self.neurons_num),
                    low=max_rate[0], high=max_rate[1])
                threshold = self.srng.uniform(
                    size=(self.array_size, self.neurons_num),
                    low=intercept[0], high=intercept[1])
                self.alpha, self.bias = theano.function(
                    [], self.neurons.make_alpha_bias(max_rates, threshold))()

                # force to 32 bit for consistency / speed
                self.bias = self.bias.astype('float64')

                # compute encoders
                self.encoders = self.make_encoders(encoders=encoders)
                # combine encoders and gain for simplification
                self.encoders = (self.encoders.T * self.alpha.T).T

            self.shared_encoders = theano.shared(self.encoders,
                name='ensemble.shared_encoders')

            # set up a dictionary for encoded_input connections
            self.encoded_input = {}
            # list of learned terminations on ensemble
            self.learned_terminations = []

            # make default origin
            self.add_origin('X', func=None, dt=dt,
                eval_points=self.eval_points, decoders=decoders)

        elif self.mode == 'direct':
            # make default origin
            self.add_origin('X', func=None, dimensions=self.dimensions*self.array_size)
            # reset neurons_num to 0
            self.neurons_num = 0

    def add_termination(self, name, pstc, decoded_input=None,
        encoded_input=None, input_socket=None, transform=None, case=None):
        """Accounts for a new termination that takes the given input
        (a theano object) and filters it with the given pstc.

        Adds its contributions to the set of decoded, encoded,
        or learn input with the same pstc. Decoded inputs
        are represented signals, encoded inputs are
        decoded_output * weight matrix, learn input is
        activities * weight_matrix.

        Can only have one of decoded OR encoded OR learn input != None.

        :param float pstc: post-synaptic time constant
        :param decoded_input:
            theano object representing the decoded output of
            the pre population (just value, not a shared variable)
        :param encoded_input:
            theano object representing the encoded output of
            the pre population multiplied by a connection weight matrix
        :param learn_input:
            theano object representing the learned output of
            the pre population multiplied by a connection weight matrix

        :param transform:
            the transform that needs to be applied (dot product) to the
            decoded output of the pre population

        :param case:
            used to generate an encoded input by applying the transform matrix
            onto the decoded pre output is a special way
        """
        # make sure one and only one of
        # (decoded_input, encoded_input) is specified
        if decoded_input is not None: assert (encoded_input is None)
        elif encoded_input is not None: assert (decoded_input is None)
        else: assert False

        if decoded_input is not None and self.mode == 'direct':
            # decoded_input is NOT the shared variable at the origin
            pre_output = theano.shared(decoded_input)
            source = TT.dot(transform, pre_output)
            self.decoded_input[name] = filter.Filter(
                name=name, pstc=pstc, source=source,
                shape=(self.array_size, self.dimensions),
                pre_output=pre_output)

        # decoded_input in this case will be the output of pre node
        elif decoded_input is not None and self.mode == 'spiking':
            # decoded_input is NOT the shared variable at the origin
            pre_output = theano.shared(decoded_input)
            source = TT.dot(transform, pre_output)
            source = TT.true_div(source, self.radius)

            self.decoded_input[name] = filter.Filter(
                name=name, pstc=pstc, source=source,
                shape=(self.array_size, self.dimensions),
                pre_output=pre_output)

        elif encoded_input is not None:
            # encoded_input is NOT the shared variable at the origin
            pre_output = theano.shared(encoded_input)
            source = case(transform, pre_output)

            self.encoded_input[name] = filter.Filter(
                name=name, pstc=pstc, source=source,
                shape=(self.array_size, self.neurons_num),
                pre_output=pre_output)

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
        raise Exception("ERROR", "Learned connections are not usable yet.")

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
        """A helper function that runs through a dictionary
        and checks for the key name, adds a digit to the end
        until a unique key has been created.

        :param string name: desired key name
        :param dict dic: the dictionary to search through
        :returns string: a unique key name for dic
        """
        i = 0
        while dic.has_key(name + '_' + str(i)):
            i += 1

        return name + '_' + str(i)

    def make_encoders(self, encoders=None):
        """Generates a set of encoders.

        :param int neurons: number of neurons
        :param int dimensions: number of dimensions
        :param theano.tensor.shared_randomstreams snrg:
            theano random number generator function
        :param list encoders:
            set of possible preferred directions of neurons

        """
        if encoders is None:
            # if no encoders specified, generate randomly
            encoders = self.srng.normal(
                (self.array_size, self.neurons_num, self.dimensions))
        else:
            # if encoders were specified, cast list as array
            encoders = np.array(encoders).T
            # repeat array until 'encoders' is the same length
            # as number of neurons in population
            encoders = np.tile(encoders,
                (self.neurons_num / len(encoders) + 1)
                               ).T[:self.neurons_num, :self.dimensions]
            encoders = np.tile(encoders, (self.array_size, 1, 1))

        # normalize encoders across represented dimensions
        norm = TT.sum(encoders * encoders, axis=[2], keepdims=True)
        encoders = encoders / TT.sqrt(norm)

        return theano.function([], encoders)()

    def make_tick(self):
        updates = OrderedDict()
        updates.update(self.update())
        self.theano_tick = theano.function([], [], updates=updates)

        # introduce 1-time-tick delay
        for o in self.origin.values():
            if o.func is not None and self.mode == 'direct': continue
            o.tick()

    def direct_mode_tick(self):
        if self.mode == 'direct':
            # set up matrix to store accumulated decoded input
            X = np.zeros((self.array_size, self.dimensions))
            # updates is an ordered dictionary of theano variables to update

            for di in self.decoded_input.values():
                # add its values to the total decoded input
                X += di.value.get_value()

            # if we're calculating a function on the decoded input
            for o in self.origin.values():
                if o.func is not None:
                    val = np.asarray([o.func(X[i]) for i in range(len(X))], dtype=np.float64)
                    o.decoded_output.set_value(val.flatten())
        else:
            raise Exception("ERROR", "The current ensemble does not have 'direct' mode.")

    # Receive the outputs of pre - decoded output - and pass it to filters
    def tick(self, inputs):
        ## Set the inputs
        for key in inputs.keys():
            val = inputs[key]
            # check if val is a decoded or an encoded input
            if self.decoded_input.has_key(key):
                self.decoded_input[key].set_pre_output(val)
            elif self.encoded_input.has_key(key):
                self.encoded_input[key].set_pre_output(val)
            else:
                raise Exception("ERROR", "Cannot identify the received input.")

        if self.mode == 'direct':
            self.direct_mode_tick()

        # should be the compiled theano function for this ensemble
        # includes the filters, ensemble, and origins updates
        self.theano_tick()

        # continue the tick in the origins
        for o in self.origin.values():
            o.tick()

    # Using the dt that was passed to the ensemble at construction time
    def update(self):
        """Compute the set of theano updates needed for this ensemble.

        Returns a dictionary with new neuron state,
        termination, and origin values.
        """

        ### find the total input current to this population of neurons

        # set up matrix to store accumulated decoded input
        X = None
        # updates is an ordered dictionary of theano variables to update
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

            # apply respective biases to neurons in the population
            J = TT.as_tensor_variable(np.array(self.bias))

            for ei in self.encoded_input.values():
                # add its values directly to the input current
                J += (ei.value.T * self.alpha.T).T
                updates.update(ei.update(self.dt))

            # only do this if there is decoded_input
            if X is not None:
                # add to input current for each neuron as
                # represented input signal x preferred direction
                J = map_gemv(1.0, self.shared_encoders, X, 1.0, J)

            # if noise has been specified for this neuron,
            if self.noise:
                # generate random noise values, one for each input_current element,
                # with standard deviation = sqrt(self.noise=std**2)
                # When simulating white noise, the noise process must be scaled by
                # sqrt(dt) instead of dt. Hence, we divide the std by sqrt(dt).
                if self.noise_type.lower() == 'gaussian':
                    J += self.srng.normal(
                        size=self.bias.shape, std=np.sqrt(self.noise/self.dt))
                elif self.noise_type.lower() == 'uniform':
                    J += self.srng.uniform(
                        size=self.bias.shape,
                        low=-self.noise/np.sqrt(self.dt),
                        high=self.noise/np.sqrt(self.dt))

            # pass that total into the neuron model to produce
            # the main theano computation
            updates.update(self.neurons.update(J, self.dt))

            for l in self.learned_terminations:
                # also update the weight matrices on learned terminations
                updates.update(l.update(self.dt))

            # and compute the decoded origin decoded_input from the neuron output
            for o in self.origin.values():
                updates.update(o.update(self.dt, updates[self.neurons.output]))

        if self.mode == 'direct':
            # if we're in direct mode then just directly pass the decoded_input
            # to the origins for decoded_output
            for o in self.origin.values():
                if o.func is None:
                    if len(self.decoded_input) > 0:
                        updates.update(OrderedDict({o.decoded_output:
                            TT.flatten(X).astype('float64')}))
        return updates

    def get_subensemble_parts(self, num_subs):
        """
        Uses encoder, decoder, alpha and bias of this ensemble and divides them
        into the specified number of parts for the subensembles.
        """
        parts = []

        decoder_parts = self.get_subensemble_decoder(num_subs, "X")

        ## We don't split on the first dimension (array_size) of these arrays,
        ## but on the second dimension (number of neurons).
        e_size = len(self.encoders[0]) / num_subs
        b_size = len(self.bias[0]) / num_subs
        a_size = len(self.alpha[0]) / num_subs

        # create the specified number of subensembles
        for e_num in range(1, num_subs + 1):
            encoder_part = []
            bias_part = []
            alpha_part = []
            for i in range(0, self.array_size):
                encoder_part.append(
                        self.encoders[i][e_size * (e_num - 1):e_size * e_num])
                bias_part.append(
                        self.bias[i][b_size * (e_num - 1):b_size * e_num])
                alpha_part.append(
                        self.alpha[i][a_size * (e_num - 1):a_size * e_num])

            parts.append((np.array(encoder_part).astype('float64'),
                np.array(decoder_parts[e_num - 1]).astype('float64'),
                np.array(bias_part).astype('float64'),
                np.array(alpha_part).astype('float64'),))

        return parts

    def get_subensemble_decoder(self, num_subs, origin_name, func=None):
        """ Gets decoder for a specified origin of the ensemble
        and divides it into the specified number of parts for subensembles
        """
        parts = []

        # create the origin in order to compute a decoder
        if origin_name not in self.origin:
            self.add_origin(origin_name, func, dt=self.dt)

        decoders = self.origin[origin_name].decoders.get_value()
        d_size = len(decoders[0]) / num_subs

        # create the specified number of decoders
        for e_num in range(1, num_subs + 1):
            decoder_part = []
            for i in range(0, self.array_size):
                decoder_part.append(
                        decoders[i][d_size * (e_num - 1):d_size * e_num])

            parts.append(np.array(decoder_part).astype('float64'))

        return parts

