from numbers import Number
import collections

import numpy as np
import theano

import zmq
import zmq_utils

class Origin(object):
    """An origin is an object that provides a signal. Origins project
    to terminations.

    This is a basic Origin, promising a set of instance variables
    to any accessing objects.
    """
    
    def __init__(self, func, dimensions=None, initial_value=None):
        """
        Either func or dimensions must be specified so that an initial 
        value can be defined.

        :param function func: the function carried out by this origin
        :param int dimensions: the number of dimensions of this origin
        :param array initial_value: the initial_value of the decoded_output
        """

        self.func = func

        if initial_value is None:
            if func is not None: 
                # initial output value = function value with input of zero(s)
                initial_value = self.func(0.0)
            elif dimensions is not None:
                initial_value = np.zeros(dimensions)
            else:
                raise Exception("Either \"func\" or \"dimensions\" must be set")

        # if scalar, make it a list
        if isinstance(initial_value, Number):
            initial_value = [initial_value]
        initial_value = np.float32(initial_value)

        # theano internal state defining output value
        self.decoded_output = theano.shared(initial_value,
            name='origin.decoded_output') 
    
        # find number of parameters of the projected value
        if dimensions is None: dimensions = len(initial_value)
        self.dimensions = dimensions

        self.output_sockets = []

    def add_output(self, output_socket_def):
        # the name is not used but may be nice to have for debugging
        name = self.func.__name__ if self.func is not None else 'X'
        self.output_sockets.append(zmq_utils.Socket(output_socket_def, name))

    def bind_sockets(self, zmq_context):
        for socket in self.output_sockets:
            socket.init(zmq_context)

    def tick(self):
        for socket in self.output_sockets:
            socket.get_instance().send_pyobj(self.decoded_output.get_value())
