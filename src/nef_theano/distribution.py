import zmq
import zmq_utils
import os
import warnings
import input

from multiprocessing import Process

DAEMON_PORT = 9000

class MixedDistributionModeException(Exception):
    pass

class LocalWorker:
    def __init__(self, zmq_context, node, name):
        self.node = node
        self.name = name
        self.zmq_context = zmq_context

        self.admin_socket_def, self.node_socket_def = \
            zmq_utils.create_ipc_socket_defs_reqrep("admin", node.name)

    def send(self, content):
        return self.admin_socket.send(content)

    def send_pyobj(self, content):
        return self.admin_socket.send_pyobj(content)

    def recv(self):
        return self.admin_socket.recv()

    def recv_pyobj(self):
        return self.admin_socket.recv_pyobj()

    def start(self, time):
        self.admin_socket = self.admin_socket_def.create_socket(self.zmq_context)
        self.process = Process(target=self.node.run,
                               args=(self.node_socket_def, time),
                               name=self.node.name)
        self.process.start()

    def stop(self):
        self.process.join()



class Worker:
    def __init__(self, name, host, worker_port, daemon_port, zmq_context):
        self.name = name

        self.zmq_context = zmq_context
        self.host = host
        self.worker_port = worker_port # specific for admin communication
        self.daemon_addr = 'tcp://%s:%s' % (host, daemon_port)
        self.worker_addr = 'tcp://%s:%s' % (host, worker_port) # admin comms

    def _communicate(self, message, addr):
        socket = self.zmq_context.socket(zmq.REQ)
        socket.connect(addr)
        message['name'] = self.name
        socket.send_pyobj(message)
        response = {'result': None}
        #  We don't want to be waiting on a response from a process that 
        #  we just told to exit because a race condition can occur where
        #  the process exits too fast and the message gets lost and
        #  causes a deadlock.
        if not (message['cmd'] == 'kill' and addr == self.worker_addr):
            response = socket.recv_pyobj()
        socket.close()
        return response['result']

    def send_command(self, message):
        is_direct = message['cmd'] == 'fin' or message['cmd'] == 'get_data'
        addr = self.worker_addr if is_direct else self.daemon_addr
        return self._communicate(message, addr)

    def kill(self):
        self._communicate(
            {'cmd': 'kill', 'args': (), 'kwargs': {}},
            self.worker_addr)
        self._communicate(
            {'cmd': 'kill', 'args': (), 'kwargs': {}},
            self.daemon_addr)

class DistributionManager:
    def __init__(self, hosts_file, job_id):
        self.workers = {}
        self.zmq_context = zmq.Context()
        self.job_id = job_id
        self.remote_hosts = None

        with open(hosts_file, 'r') as f:
            self.remote_hosts = [host.strip() for host in f.readlines()]
            self.next_host_id = 0

            # Tests all hosts for availability, and locks all those that are available
            self.remote_hosts = \
                [host for host in self.remote_hosts if self.lock(host)]

            if not len(self.remote_hosts):
                print("DEBUG: All hosts are busy with other jobs. Try again later...")
                exit(1)

    def __del__(self):
        for host in self.remote_hosts:
            self.unlock(host)

    def send_usr_module(self, module_name, module_contents):
        for host in self.remote_hosts:
            try:
                self._send_message_to_daemon({
                    'cmd': 'write_usr_module',
                    'name': None,
                    'args': (module_name, module_contents),
                    'kwargs': {}},
                    "tcp://%s:%s" % (host, DAEMON_PORT))
            except zmq.ZMQError:
                continue

    def lock(self, host):
        response = self._send_message_to_daemon(
            {'cmd': 'lock', 'name': None, 'args': (self.job_id, ), 'kwargs': {}},
            "tcp://%s:%s" % (host, DAEMON_PORT))

        return response == 'ack'

    def unlock(self, host):
        response = self._send_message_to_daemon(
            {'cmd': 'unlock', 'name': None, 'args': (self.job_id, ), 'kwargs': {}},
            "tcp://%s:%s" % (host, DAEMON_PORT))

        return response == 'ack'

    def _next_host(self):
        running_hosts = len(self.remote_hosts)

        while True:
            host = self.remote_hosts[self.next_host_id]
            self.next_host_id += 1

            if self.next_host_id > len(self.remote_hosts) - 1:
                self.next_host_id = 0
            try:
                # See if the host is alive. If not, try the next one
                response = self._send_message_to_daemon(
                    {'cmd': 'ping', 'name': None, 'args': (), 'kwargs': {}},
                    "tcp://%s:%s" % (host, DAEMON_PORT))

                if response != 'pong':
                    print "DEBUG: Host %s:%s is busy."
                else:
                    break
            except zmq.ZMQError:
                print "DEBUG: Host %s:%s is down." % (host, DAEMON_PORT)

            running_hosts -= 1
            if running_hosts == 0:
                raise Exception("ERROR: All hosts are down. Make sure that " +
                                "Spike daemons on the remote hosts are running and available.")
                exit(1)

        return host

    def _send_message_to_daemon(self, message, daemon_addr):
        socket = self.zmq_context.socket(zmq.REQ)
        poller = zmq.Poller()

        poller.register(socket, zmq.POLLIN)
        socket.connect(daemon_addr)
        socket.send_pyobj(message, zmq.NOBLOCK)

        response = None
        responses = dict(poller.poll(60000)) # 1 minute

        if not (socket in responses and responses[socket] == zmq.POLLIN):
            warnings.warn("WARNING: We have been waiting for over a minute for machine '" + daemon_addr + "' it might be down (or computing a very large node)")
            responses = dict(poller.poll()) # forever

        response = socket.recv_pyobj(zmq.NOBLOCK)

        socket.close()

        if response == None:
            raise zmq.ZMQError()
        return response['result']

    def _new_daemon_port(self, daemon_addr, name):
        return self._send_message_to_daemon(
            {'cmd': 'next_avail_port', 'name': name, 'args': (), 'kwargs': {}},
            daemon_addr)

    def new_worker(self, name, local=False, node=None):
        if local:
            return LocalWorker(self.zmq_context, node, name)

        host_name = self._next_host()
        daemon_addr = "tcp://%s:%s" % (host_name, DAEMON_PORT)
        worker_port = self._new_daemon_port(daemon_addr, name)

        worker = Worker(name, host_name, worker_port, DAEMON_PORT, self.zmq_context)
        return worker

