import zmq
from nef_theano import input, ensemble, probe
from nef_theano import connection, zmq_utils

from multiprocessing import Process

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

        if transform is not None:
            raise Exception("ERROR: not expecting a transform")

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

    def connect_pre(self, worker_name, post_addr):
        wnode = self.workers[worker_name]['node']
        worigin = wnode.origin['X']
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
