import collections

import numpy as np
import theano
import collections

import numpy as np
import theano
import theano.tensor as TT

from .filter import Filter

from multiprocessing import Process
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

    def run(self, ticker_socket_def):
        self.__bind_sockets()
        ticker_conn = ticker_socket_def.create_socket(self.zmq_context)

        sim_time = float(ticker_conn.recv())

        for i in range(int(sim_time / self.dt)):
            self.t = self.run_time + i * self.dt
            self.tick()

        self.run_time += sim_time

        ticker_conn.send("FIN") # inform main proc that probe finished
        ticker_conn.recv() # wait for an ACK from main proc before finishing

        # send all recorded data to the administrator
        data = self.data[:self.i+1]
        ticker_conn.send_pyobj(data)
        ticker_conn.recv() # want an ack of receiving the data

    def __bind_sockets(self):
        # create a context for this probe process if do not have one already
        if self.zmq_context is None:
            self.zmq_context = zmq.Context()

        # create socket connections for the input socket
        self.input_socket.init(self.zmq_context)

    def add_input(self, input_socket):
        # probes have a one to one mapping with a target ensembles
        # thus will have just one input and can use the probe name for the input socket
        self.input_socket = zmq_utils.Socket(input_socket, self.name)

class AggregatorProbe(object):
    """ A class to aggregate the output of multiple probes """

    def __init__(self, name, net):
        self.name = name
        self.net = net

        self.probes = {}

        # context should be created when the process is started (bind_sockets)
        self.zmq_context = None
        self.input_socket = None

    def add(self, probe):
        # add probe to the aggregator
        self.probes[probe.target_name] = {
            "probe": probe,
            "data": []
        }

    def get_data(self):
        # access the data for this node, which is stored in the network
        return self.net.get_probe_data(self.name)

    def run(self, ticker_socket_def):
        # create a context for this probe process if do not have one already
        if self.zmq_context is None:
            self.zmq_context = zmq.Context()

        ticker_conn = ticker_socket_def.create_socket(self.zmq_context)

        # start the probe processes
        for probe_name in self.probes:
            probe = self.probes[probe_name]["probe"]

            # connect the aggregator probe with the probe
            aggregator_socket, probe_socket = \
                zmq_utils.create_socket_defs_reqrep("aggregator", probe_name)

            proc = Process(target=probe.run, args=(probe_socket,), name=probe_name)
            self.probes[probe_name]["connection"] = \
                zmq_utils.Socket(aggregator_socket, self.name).init(self.zmq_context)
            proc.start()

        # receive start time from network
        sim_time = float(ticker_conn.recv())
        # forward start time to probes
        for probe in self.probes.keys():
            self.probes[probe]["connection"].send(str(sim_time))

        # wait until all probes are done
        for probe in self.probes.keys():
            probe_conn = self.probes[probe]["connection"]
            probe_conn.recv()
            probe_conn.send("ACK")

        # tell admin that probes are done receiving data
        ticker_conn.send("FIN")
        ticker_conn.recv()

        # wait for data from probes
        for probe in self.probes.keys():
            probe_conn = self.probes[probe]["connection"]
            self.probes[probe]["data"] = probe_conn.recv_pyobj()
            probe_conn.send("ACK")

        # sum up all the data received from the probes
        np.set_printoptions(precision=15)
        data = None
        for probe in self.probes.keys():
            probe_data = self.probes[probe]["data"]
            if data is None:
                data = probe_data
            else:
                #print "type is :",type(data[0][0])
                #print "Adding A:",data
                #print "type is :",type(probe_data[0][0])
                #print "To B:",probe_data
                data += probe_data

        # send final data to the network
        ticker_conn.send_pyobj(data)
        ticker_conn.recv() # wait for an ack from the network
