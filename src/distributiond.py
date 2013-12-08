import zmq
from nef_theano import input, ensemble, probe
from nef_theano import connection, zmq_utils

import sys, tempfile, os
tempdir = tempfile.gettempdir()
sys.path.append(tempdir)

from multiprocessing import Process
import numpy as np

# Distributor listens on this port for commands
LISTENER_ENDPOINT = "tcp://*:9000"
WORKER_PORT = 8000

class DistributionDaemon:
    def __init__(self):
        self.zmq_context = zmq.Context()
        self.listener_socket = None

        self.workers = {} # format => {name: {node: obj, proc: obj}}
        self.current_port_offset = 0

    def _get_next_port(self):
        next_port = WORKER_PORT + self.current_port_offset
        self.current_port_offset += 1

        # Roll over after 1000 port assignments
        if self.current_port_offset >= 1000:
            self.current_port_offset = 0

        return next_port

    def ping(self, worker_name):
        self.listener_socket.send_pyobj({'result': 'pong'})

    def write_usr_module(self, worker_name, module_name, module_contents):
        file_path = os.path.join(tempdir, module_name)
        with open(file_path, 'w') as module_file:
            module_file.write(module_contents)
        self.listener_socket.send_pyobj({'result': 'ack'})

    def next_avail_port(self, worker_name):
        next_port = self._get_next_port()
        print "DEBUG: Next available port: {port}".format(port=next_port)
        self.listener_socket.send_pyobj({'result': str(next_port)})

    def make_input(self, worker_name, *args, **kwargs):
        inode = input.Input(*args, **kwargs)
        self.workers[worker_name] = {'node': inode}
        self.listener_socket.send_pyobj({'result': 'ack'})

    def make_ensemble(self, worker_name, *args, **kwargs):
        enode = ensemble.EnsembleProcess(*args, **kwargs)
        self.workers[worker_name] = {'node': enode}
        self.listener_socket.send_pyobj({'result': 'ack'})

    def make_probe(self, worker_name, wport, *args, **kwargs):
        pnode = probe.Probe(*args, **kwargs)
        self.workers[worker_name] = {'node': pnode}

        socket = zmq_utils.SocketDefinition(
            "tcp://*:%s" % (wport), zmq.PULL, is_server=True)
        pnode.add_input(socket)

        self.listener_socket.send_pyobj({'result': 'ack'})

    def _connect(self, post_node, post_socket, pre_name, pre_output, pre_dimensions,
        transform=None, weight=1, index_pre=None, index_post=None, pstc=0.01,
        func=None):
        post = post_node

        if transform is not None:
            assert ((weight == 1) and (index_pre is None) and (index_post is None))

            transform = np.array(transform)

            if transform.shape[0] != post.dimensions * post.array_size or len(transform.shape) > 2:

                if transform.shape[0] == post.array_size * post.neurons_num:
                    transform = transform.reshape(
                                      [post.array_size, post.neurons_num] +\
                                                list(transform.shape[1:]))

                if len(transform.shape) == 2: # repeat array_size times
                    transform = np.tile(transform, (post.array_size, 1, 1))

                # check for pre side encoded connection (case 3)
                if len(transform.shape) > 3 or \
                       transform.shape[2] == pre.array_size * pre.neurons_num:

                    raise Exception("ERROR", "connection 3 not handled yet")

                    if transform.shape[2] == pre.array_size * pre.neurons_num:
                        transform = transform.reshape(
                                        [post.array_size, post.neurons_num,
                                              pre.array_size, pre.neurons_num])
                    assert transform.shape == \
                            (post.array_size, post.neurons_num,
                             pre.array_size, pre.neurons_num)

                    print 'setting pre_output=spikes'

                    # get spiking output from pre population
                    pre_output = pre.neurons.output

                    case1 = connection.Case1(
                        (post.array_size, post.neurons_num,
                         pre.array_size, pre.neurons_num))

                    # pass in the pre population decoded output value
                    # to the post population
                    post.add_termination(input_socket=destination_socket,
                        name=pre_name, pstc=pstc,
                        encoded_input= pre_output.get_value(),
                        transform=transform, case=case1)

                    pre_origin.add_output(origin_socket)

                    return

                else: # otherwise we're in case 2 (pre is decoded)
                    raise Exception("ERROR", "connection 2 not handled yet")

                    assert transform.shape ==  \
                               (post.array_size, post.neurons_num, pre_dimensions)

                    # can't specify a function with either side encoded connection
                    assert func == None

                    case2 = connection.Case2(
                        (post.array_size, post.neurons_num,
                         pre.array_size, pre.neurons_num))

                    # pass in the pre population decoded output value
                    # to the post population
                    post.add_termination(input_socket=destination_socket,
                        name=pre_name, pstc=pstc,
                        encoded_input= pre_output.get_value(),
                        transform=transform, case=case2)

                    pre_origin.add_output(origin_socket)

                    return

        transform = connection.compute_transform(
            dim_pre=pre_dimensions,
            dim_post=post_node.dimensions,
            array_size=post_node.array_size,
            weight=weight,
            index_pre=index_pre,
            index_post=index_post,
            transform=transform)

        post_node.add_termination(input_socket=post_socket, name=pre_name,
            pstc=pstc, decoded_input=pre_output, transform=transform)

    def connect_pre(self, worker_name, post_addr, **kwargs):
        wnode = self.workers[worker_name]['node']
        func = kwargs.pop('func', None)
        decoders = kwargs.pop('decoders', None)
        is_subensemble = isinstance(wnode, ensemble.EnsembleProcess) and wnode.ensemble.is_subensemble

        if func is None:
            worigin = wnode.origin['X']
        else:
            origin_name = func.__name__
            if origin_name not in wnode.origin:
                if is_subensemble:
                    wnode.add_origin(origin_name, None, decoders=decoders, **kwargs)
                else:
                    wnode.add_origin(origin_name, func, **kwargs)
            worigin = wnode.origin[origin_name]

        worigin_output = worigin.decoded_output.get_value()
        worigin_dimensions = worigin.dimensions

        # connect worigin to post
        socket = zmq_utils.SocketDefinition(post_addr, zmq.PUSH, is_server=False)
        worigin.add_output(socket)

        self.listener_socket.send_pyobj({
            'result' : {
                'pre_output': worigin_output,
                'pre_dimensions': worigin_dimensions
            }
        })

    def connect_post(self, worker_name, *args, **kwargs):
        worker = self.workers[worker_name]
        wnode = worker['node']
        wport = kwargs.pop('post_port')

        pre_name = kwargs.pop('pre_name')
        pre_output = kwargs.pop('pre_output')
        pre_dimensions = kwargs.pop('pre_dimensions')
        socket = zmq_utils.SocketDefinition(
            "tcp://*:%s" % (wport), zmq.PULL, is_server=True)

        self._connect(wnode, socket, pre_name, pre_output, pre_dimensions,
            *args, **kwargs)

        self.listener_socket.send_pyobj({'result': 'ack'})

    def start(self, worker_name, admin_port, time):
        worker = self.workers[worker_name]
        wnode = worker['node']

        admin_socket = zmq_utils.SocketDefinition(
            "tcp://*:%s" % (admin_port), zmq.REP, is_server=True)

        process = Process(target=wnode.run, args=(admin_socket, time), name=wnode.name)
        process.start()
        worker['proc'] = process

        self.listener_socket.send_pyobj({'result': 'ack'})

    def kill(self, worker_name):
        self.workers[worker_name]['proc'].join()
        self.workers.pop(worker_name)
        self.listener_socket.send_pyobj({'result': 'ack'})

    def listen(self, endpoint):
        self.listener_socket = self.zmq_context.socket(zmq.REP)
        self.listener_socket.bind(endpoint)

        while True:
            msg = self.listener_socket.recv_pyobj()
            print "DEBUG: Received Message: {msg}".format(msg=msg)

            cmd = msg['cmd']
            worker_name = msg['name']
            args = msg['args']
            kwargs = msg['kwargs']
            # cmd is a command that corresponds to a method in the daemon
            # invoke the cmd methods with given arguments
            getattr(self, cmd)(worker_name, *args, **kwargs)

def main():
    DistributionDaemon().listen(LISTENER_ENDPOINT)

if __name__ == '__main__':
    main()
