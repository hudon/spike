import zmq
import zmq_utils
import os

from multiprocessing import Process

class Worker:
    def __init__(self, zmq_context, node, is_distributed, host, worker_port,
            daemon_port):
        self.node = node
        self.zmq_context = zmq_context
        self.is_distributed = is_distributed

        if is_distributed:
            self.daemon_host = 'tcp://%s:%s' % (host, daemon_port)
            self.worker_port = worker_port
            self.admin_socket_def, self.node_socket_def = \
                  zmq_utils.create_tcp_socket_defs_reqrep(host, worker_port)
        else:
            self.admin_socket_def, node_socket_def = \
                zmq_utils.create_ipc_socket_defs_reqrep("admin", node.name)
            self.process = Process(target=node.run, args=(node_socket_def,), name=node.name)

    def send(self, content):
        return self.admin_socket.send(content)

    def recv(self):
        return self.admin_socket.recv()

    def recv_pyobj(self):
        return self.admin_socket.recv_pyobj()

    def start(self):
        self.admin_socket = self.admin_socket_def.create_socket(self.zmq_context)

        if self.is_distributed:
            socket = self.zmq_context.socket(zmq.REQ)
            socket.connect(self.daemon_host)

            #NOTE: send functions bound at runtime by putting them in a
            # functions.py file and sending that
            nef_dir = os.path.dirname(__file__)
            funcs_file = os.path.join(nef_dir, "../functions.py")
            with open(funcs_file, "rb") as funcs:
                socket.send_pyobj({
                    "node": self.node,
                    "socket": self.node_socket_def,
                    "functions": funcs.read()
                })
            socket.recv() # wait for an ACK from the daemon
            socket.close()
        else:
            self.process.start()

    def stop(self):
        if self.is_distributed:
            socket = self.zmq_context.socket(zmq.REQ)
            socket.connect(self.daemon_host)
            socket.send_pyobj(('FIN', self.node.name))
            socket.recv() # wait for an ACK from the daemon
            socket.close()
        else:
            self.process.join()


class DistributionManager:
    """ Class responsible for socket creation and work distribution """

    def __init__(self, is_distributed=False):
        self.workers = {}
        if is_distributed:
            self.worker_port = 8000
            self.current_port_offset = 1

        self.zmq_context = zmq.Context()
        self.is_distributed = is_distributed

    def create_worker(self, node):
        if self.is_distributed:
            worker_port = self._new_port(self.worker_port)
            worker = Worker(self.zmq_context, node, self.is_distributed,
                    'localhost', worker_port, 9000)
        else:
            worker = Worker(self.zmq_context, node, self.is_distributed)
        self.workers[node.name] = worker
        return worker

    def _new_port(self, starter_port):
        p = str(starter_port + self.current_port_offset)
        self.current_port_offset += 1
        return p

    def connect(self, src_name, dst_name):
        """ Doesn't actually connect yet but provides two sockets definitions
        """
        if self.is_distributed:
            port = self._new_port(self.worker_port)
            return zmq_utils.create_tcp_socket_defs_pushpull('localhost', port)
        return zmq_utils.create_ipc_socket_defs_pushpull(src_name, dst_name)
