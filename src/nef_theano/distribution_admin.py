import zmq
from . import zmq_utils

class DistributedWorker:
    def __init__(self, dist_admin, node, host):
        self.dist_admin = dist_admin
        self.node = node
        self.host = host

    def start(self):
        self.dist_admin.start_worker(self.node.name, self)

    def stop(self):
        self.dist_admin.stop_worker(self.node.name)

class DistributionAdmin:
    """ Class responsible for socket creation and work distribution """

    def __init__(self, host_list, distribute=False):
        self.last_port = 10010
        self.zmq_context = zmq.Context()
        self.workers = {}
        self.host_list = host_list
        self.distribute = distribute

    def create_reqrep_pair(self, src_name, dest_node):
        """ 
        Create a pair of socket definitions for a REQ/REP connection.
        Origin TCP sockets always include the IP of the destination, plus an allocated port.
        Destination TCP sockets are always 'localhost:<allocated port>'.
        An endpoint worker definition is generated to maintain a reference to the soon-to-be allocated worker.

        Returns a tuple: (origin_socket_defn, destination_socket_defn, destination_worker_defn)
        """
        if self.distribute:
            self.last_port += 1
            port = self.last_port
            host = None
            worker = None
            if dest_node.name in self.workers:
                worker = self.workers[dest_node.name]
                host = worker.host
            else:
                host = self._find_host()
                worker = DistributedWorker(self, dest_node, host)
                self.workers[dest_node.name] = worker

            destination_bind_address = "%s:%s" % (host, port)

            return zmq_utils.create_tcp_socket_defs_reqrep(destination_bind_address, "*:%s" % port) + (worker,)
        else:
            return zmq_utils.create_ipc_socket_defs_reqrep(src_name, dest_node.name) + (DistributedWorker(self, dest_node, ""),)

    def create_pushpull_pair(self, src_name, dest_node):
        """ 
        Create a pair of socket definitions for a PUSH/PULL connection.
        Origin TCP sockets always include the IP of the destination, plus an allocated port.
        Destination TCP sockets are always 'localhost:<allocated port>'.
        An endpoint worker definition is generated to maintain a reference to the soon-to-be allocated worker.

        Returns a tuple: (origin_socket_defn, destination_socket_defn, destination_worker_defn)
        """
        if self.distribute:
            self.last_port += 1
            port = self.last_port
            host = None
            worker = None
            if dest_node.name in self.workers:
                worker = self.workers[dest_node.name]
                host = worker.host
            else:
                host = self._find_host()
                worker = DistributedWorker(self, dest_node, host)
                self.workers[dest_node.name] = worker

            destination_bind_address = "%s:%s" % (host, port)

            return zmq_utils.create_tcp_socket_defs_pushpull(destination_bind_address, "*:%s" % port) + (worker,)
        else:
            return zmq_utils.create_ipc_socket_defs_pushpull(src_name, dest_node.name) + (DistributedWorker(self, dest_node, ""),)
        
    def start_worker(self, name, worker):
        if self.distribute:
            self._send_object_to_endpoint(worker.node, "tcp://%s:10010" % worker.host)
        else:
            Process(target=worker.node.run).start()
            
    def stop_worker(self, name):
        pass

    def _find_host(self):
        #TODO: return the next available host (round-robin would work)
        return self.host_list[0]

    def _send_object_to_endpoint(self, obj_to_send, endpoint):
        response = None

        socket = self.zmq_context.socket(zmq.REQ)
        socket.connect(endpoint)

        poller = zmq.Poller()
        poller.register(socket, zmq.POLLIN)
        print "Sending object to distributor at %s: %s" % (endpoint, obj_to_send)
        socket.send_pyobj(obj_to_send)

        if poller.poll(10000): #10s timeout
            socket.recv()
        else:
            # Handle no-response scenario
            pass

        socket.close()

        return response
