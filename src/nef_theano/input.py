from numbers import Number

import theano
from theano import tensor as TT
import numpy as np

from . import origin

import zmq
import zmq_utils

class Input(object):
    """Inputs are objects that provide real-valued input to ensembles.

    Any callable can be used an input function.

    """
    def __init__(self, name, value, zero_after_time=None, is_printing=False):
        """
        :param string name: name of the function input
        :param value: defines the output decoded_output
        :type value: float or function
        :param float zero_after_time:
            time after which to set function output = 0 (s)
        :param bool is_printing: should the process be printing values to stdout
        """
        self.name = name
        self.is_printing = is_printing
        self.t = 0
        self.function = None
        self.zero_after_time = zero_after_time
        self.zeroed = False
        self.change_time = None
        self.origin = {}

        # context should be created when the process is started (bind_sockets)
        self.zmq_context = None

        # if value parameter is a python function
        if callable(value): 
            self.origin['X'] = origin.Origin(func=value)
        # if value is dict of time:value pairs
        elif isinstance(value, dict):
            self.change_time = sorted(value.keys())[0]
            # check for size of dict elements
            if isinstance(value[self.change_time], list):
                initial_value = np.zeros(len(value[self.change_time]))
            else: initial_value = np.zeros(1)
            self.origin['X'] = origin.Origin(func=None, 
                initial_value=initial_value)
            self.values = value
        else:
            self.origin['X'] = origin.Origin(func=None, initial_value=value)

    def reset(self):
        """Resets the function output state values.
        
        """
        self.zeroed = False

    def tick(self):
        """Move function input forward in time."""
        if self.zeroed:
            return

        # zero output
        if self.zero_after_time is not None and self.t > self.zero_after_time:
            self.origin['X'].decoded_output.set_value(
                np.float32(np.zeros(self.origin['X'].dimensions)))
            self.zeroed = True
    
        # change value
        if self.change_time is not None and self.t > self.change_time:
            self.origin['X'].decoded_output.set_value(
                np.float32(np.array([self.values[self.change_time]])))
            index = sorted(self.values.keys()).index(self.change_time) 
            if index < len(self.values) - 1:
                self.change_time = sorted(self.values.keys())[index+1]
            else: self.change_time = None

        # update output decoded_output
        if self.origin['X'].func is not None:
            value = self.origin['X'].func(self.t)
            # if value is a scalar output, make it a list
            if isinstance(value, Number):
                value = [value] 

            # cast as float32 for consistency / speed,
            # but _after_ it's been made a list
            self.origin['X'].decoded_output.set_value(np.float32(value)) 

        for o in self.origin.values():
            o.tick()

    def run(self, ticker_socket_def):
        self.bind_sockets()
        ticker_conn = ticker_socket_def.create_socket(self.zmq_context)

        while True:
            msg = ticker_conn.recv()

            if msg == "END":
                break

            self.t = float(msg)
            self.tick()

            if self.is_printing:
                print self.origin['X'].decoded_output.get_value()

            ticker_conn.send("")

    def bind_sockets(self):
        # create a context for this ensemble process if do not have one already
        if self.zmq_context is None:
            self.zmq_context = zmq.Context()

        for o in self.origin.values():
            o.bind_sockets(self.zmq_context)

