from theano import tensor as TT
import theano
import numpy

class Input:
    def __init__(self, name, value, zero_after=None):
        self.name = name
        self.t = 0
        self.function = None
        self.zero_after = zero_after
        self.zeroed = False

        self.output_pipes = []

        if callable(value):
            v = value(0.0)
            self.value = numpy.array(v).astype('float32')
            self.function = value
        else:
            self.value = numpy.array(value).astype('float32')

    def add_output(self, output):
        self.output_pipes.append(output)

    def tick(self):
        if self.zeroed: return

        if self.zero_after is not None and self.t > self.zero_after:
            self.value = numpy.zeros_like(self.value)
            self.zeroed = True

        if self.function is not None:
            self.value = self.function(self.t)

        for pipe in self.output_pipes:
            pipe.send(self.value)

    def reset(self):
        self.zeroed = False

    def run(self, ticker_conn):
        while True:
            ticker_conn.recv()
            self.tick()
            ticker_conn.send(1)
