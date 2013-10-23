import collections

import numpy as np
import theano
import collections

import numpy as np
import theano
import theano.tensor as TT

from .filter import Filter

import zmq
import zmq_utils

class Probe(object):
    """A class to record from things (i.e., origins).

    """
    buffer_size = 1000

    def __init__(self, name, target, target_name, dt_sample, dt, net, pstc=0.03):
        """
        :param string name:
        :param target:
        :type target:
        :param string target_name:
        :param float dt_sample:
        :param float pstc:
        """
        self.name = name
        self.target = theano.shared(target)
        self.target_name = target_name
        self.dt_sample = dt_sample
        self.dt = dt
        self.run_time = 0.0
        self.net = net

        # context should be created when the process is started (bind_sockets)
        self.zmq_context = None
        self.input_socket = None

        # create array to store the data over many time steps
        self.data = np.zeros((self.buffer_size,) + self.target.get_value().shape)
        self.i = -1 # index of the last sample taken

        # create a filter to filter the data
        self.filter = Filter(name=name, pstc=pstc, source=self.target)
        updates = {}
        updates.update(self.update(self.dt))
        self.theano_tick = theano.function([], [], updates=updates)

    def update(self, dt):
        """
        :param float dt: the timestep of the update
        """
        return self.filter.update(dt)

    def tick(self):
        # Wait to receive the origin output from the ensemble
        val = self.input_socket.get_instance().recv_pyobj()
        self.target.set_value(val)

        # Filter and store the received output value
        i_samp = int(round(self.t / self.dt_sample, 5))

        if i_samp > self.i:
            if i_samp >= len(self.data):
                # increase the buffer
                self.data = np.vstack(
                    [self.data, np.zeros((self.buffer_size,)
                                         + self.data.shape[1:])])

            # record the filtered value
            self.data[self.i+1:i_samp+1] = self.filter.value.get_value()
            self.i = i_samp

        self.theano_tick()

    def get_data(self):
        # access the data for this node, which is stored in the network
        return self.net.get_probe_data(self.name)

    def run(self, admin_socket_def):
        self.bind_sockets()
        admin_conn = admin_socket_def.create_socket(self.zmq_context)

        sim_time = float(admin_conn.recv())

        for i in range(int(sim_time / self.dt)):
            self.t = self.run_time + i * self.dt
            self.tick()

        self.run_time += sim_time

        admin_conn.send("FIN") # inform main proc that probe finished
        admin_conn.recv() # wait for an ACK from main proc before finishing

        # send all recorded data to the administrator
        data = self.data[:self.i+1]
        admin_conn.send_pyobj(data)
        admin_conn.recv() # want an ack of receiving the data

    def bind_sockets(self):
        # create a context for this probe process if do not have one already
        if self.zmq_context is None:
            self.zmq_context = zmq.Context()

        # create socket connections for the input socket
        self.input_socket.init(self.zmq_context)

    def add_input(self, input_socket):
        # probes have a one to one mapping with a target ensembles
        # thus will have just one input and can use the probe name for the input socket
        self.input_socket = zmq_utils.Socket(input_socket, self.name)
