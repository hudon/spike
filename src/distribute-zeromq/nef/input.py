from theano import tensor as TT
import theano
import numpy
import zmq
import zmq_utils

class Input:
    def __init__(self, name, value, zero_after=None):
        self.name = name
        self.t = 0
        self.function = None
        self.zero_after = zero_after
        self.zeroed = False

        self.output_socket_definitions = []
        self.output_sockets = []
        self.ticker_conn = None

        if callable(value):
            v = value(0.0)
            self.value = numpy.array(v).astype('float32')
            self.function = value
        else:
            self.value = numpy.array(value).astype('float32')

    def __del__(self):
        for socket in self.output_sockets:
            socket.close()

    def add_output(self, output_definition):
        self.output_socket_definitions.append(output_definition)

    def tick(self):
        if self.zeroed: return

        if self.zero_after is not None and self.t > self.zero_after:
            self.value = numpy.zeros_like(self.value)
            self.zeroed = True

        if self.function is not None:
            self.value = self.function(self.t)

        for socket in self.output_sockets:
            socket.send_pyobj(self.value)

    def reset(self):
        self.zeroed = False

    def bind_sockets(self):
       for defn in self.output_socket_definitions:
           self.output_sockets.append(defn.create_socket())

        # zmq.REP strictly enforces alternating recv/send ordering
       zmq_context = zmq.Context()
       self.ticker_conn = zmq_context.socket(zmq.REP)
       self.ticker_conn.connect(zmq_utils.TICKER_SOCKET_LOCAL_NAME)

    def run(self):
        self.bind_sockets()

        while True:
            self.t = float(self.ticker_conn.recv())
            self.tick()
            self.ticker_conn.send("")
