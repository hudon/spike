import zmq
import zmq_utils

from multiprocessing import Process

DAEMON_PORT = 9000

class Worker:
    def __init__(self, zmq_context, node, is_distributed, 
                 host=None, worker_port=None, daemon_port=DAEMON_PORT):
        self.node = node
        self.zmq_context = zmq_context
        self.is_distributed = is_distributed

        if is_distributed:
            self.host_name = host
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
            socket.send_pyobj({
                "node": self.node,
                "socket": self.node_socket_def
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

    def __init__(self, is_distributed=False, hosts_file=None):
        self.workers = {}

        self.zmq_context = zmq.Context()
        self.is_distributed = is_distributed

        if is_distributed:
            self.remote_hosts = None
            
            if hosts_file:
                with open(hosts_file, 'r') as f:
                    self.remote_hosts = [host.strip() for host in f.readlines()]
                    self.next_host_id = 0
            else:
                print "ERROR: (DistributionManager) Hosts file not specified for distributed simulation"
                exit(1)

    def create_worker(self, node):
        if self.is_distributed:
            host_name = self._next_host()
            daemon_host = "tcp://%s:%s" % (host_name, DAEMON_PORT)

            worker_port = self._new_daemon_port(daemon_host)

            worker = Worker(self.zmq_context, node, self.is_distributed,
                            host_name, worker_port, DAEMON_PORT)
        else:
            worker = Worker(self.zmq_context, node, self.is_distributed)
        self.workers[node.name] = worker
        return worker

    def _next_host(self):
        if not self.is_distributed:
            return None
        
        running_hosts = len(self.remote_hosts)

        while True:
            host = self.remote_hosts[self.next_host_id]

            self.next_host_id += 1
            if self.next_host_id > len(self.remote_hosts) - 1:
                self.next_host_id = 0
            try:
                # See if the host is alive. If not, try the next one
                self._send_message_to_daemon("PING", "tcp://%s:%s" % (host, DAEMON_PORT))
                break
            except zmq.ZMQError:
                print "DEBUG: Host %s:%s is down." % (host, DAEMON_PORT)

                running_hosts -= 1
                if running_hosts == 0:
                    print "ERROR: All hosts are down. Make sure that " + \
                        "Spike daemons are running on the remote hosts."
                    exit(1)
                pass

        return host

    def _send_message_to_daemon(self, message, daemon_host):
        socket = self.zmq_context.socket(zmq.REQ)
        socket.connect(daemon_host)
        socket.send_pyobj(message, zmq.NOBLOCK)

        response = socket.recv(zmq.NOBLOCK) # Get port from daemon

        socket.close()        

        return response

    def _new_daemon_port(self, daemon_host):
        return self._send_message_to_daemon('NEXT_AVAIL_PORT', daemon_host)

    def connect(self, src_name, dst_name):
        """ Doesn't actually connect, but provides two sockets definitions
        """
        if self.is_distributed:
            dst_worker = self.workers[dst_name]

            port = self._new_daemon_port(dst_worker.daemon_host)
            dest_host = dst_worker.host_name

            return zmq_utils.create_tcp_socket_defs_pushpull(dest_host, port)
        return zmq_utils.create_ipc_socket_defs_pushpull(src_name, dst_name)
