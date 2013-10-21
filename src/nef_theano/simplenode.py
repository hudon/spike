from numbers import Number
import inspect

import numpy as np
import theano

from . import origin

import zmq
import zmq_utils

class SimpleNode(object):
    """A SimpleNode allows you to put arbitary code as part of an NEF model.

    This object has Origins and Terminations which can be used just like
    any other Nengo component. Arbitrary code can be run every time step,
    making this useful for simulating sensory systems (reading data
    from a file or a webcam, for example), motor systems (writing data to
    a file or driving a robot, for example), or even parts of the brain
    that we don't want a full neural model for (symbolic reasoning or
    declarative memory, for example).

    You can have as many origins you like.  The dimensionality
    of the origins are set by the length of the returned vector of floats.

      class SquaringFiveValues(nef.SimpleNode):
          def init(self):
              self.value=0
          def origin_output(self):
              return [self.value]

    There is also a special method called tick() that is called once per
    time step.

      class HelloNode(nef.SimpleNode):
          def tick(self):
              print 'Hello world'

    The current time can be accessed via `self.t`.  This value will be the
    time for the beginning of the current time step.  The end of the current
    time step is `self.t_end`.

    """

    def __init__(self, name, dt=.001):
        """
        :param string name: the name of the created node
        :param float pstc: the default time constant on the filtered inputs
        :param int dimensions:
            the number of dimensions of the decoded input signal
        """
        self.name = name
        self.t = 0  # current simulation time
        self.dt = dt
        self.run_time = 0

        self.origin = {}

        # context should be created when the process is started (bind_sockets)
        self.zmq_context = None

        self.init()  # initialize internal variables if there are any

        # look at all the defined methods, if any start with 'origin_',
        # make origins that implement the defined function
        for name, method in inspect.getmembers(self, inspect.ismethod):
            if name.startswith('origin_'):
                # add to dictionary of origins
                self.origin[name[7:]] = origin.Origin(
                    func=method, initial_value=method())

    def init(self):
        """Initialize the node.

        Override this to initialize any internal variables. This will
        also be called whenever the simulation is reset.

        """
        pass

    def tick(self):
        """An extra utility function that is called every time step.

        Override this to create custom behaviour that isn't necessarily tied
        to a particular input or output. Often used to write spike data
        to a file or produce some other sort of custom effect.

        """
        pass

    def reset(self, **kwargs):
        """Reset the state of all the internal variables."""
        self.init(**kwargs)

    def main_tick(self):
        """Run the simple node.

        :param float start: The time to start running
        :param float end: The time to stop running

        """
        self.tick()

        for origin in self.origin.values():
            value = origin.func()

            # if value is a scalar output, make it a list
            if isinstance(value, Number):
                value = [value]

            # cast as float32 for consistency / speed,
            # but _after_ it's been made a list
            origin.decoded_output.set_value(np.float32(value))
            origin.tick()

    def run(self, ticker_socket_def):
        self.bind_sockets()
        ticker_conn = ticker_socket_def.create_socket(self.zmq_context)

        sim_time = float(ticker_conn.recv())

        for i in range(int(sim_time / self.dt)):
            self.t = self.run_time + i * self.dt
            self.main_tick()

        self.run_time += sim_time

        ticker_conn.send("FIN") # inform main proc that node finished
        ticker_conn.recv() # wait for an ACK from main proc before exiting

    def bind_sockets(self):
        # create a context for this input process if do not have one already
        if self.zmq_context is None:
            self.zmq_context = zmq.Context()

        for o in self.origin.values():
            o.bind_sockets(self.zmq_context)
