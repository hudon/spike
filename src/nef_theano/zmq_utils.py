import zmq

# Straight-up IPC connection between processes (Does NOT work on Windows)
TICKER_SOCKET_LOCAL_NAME = "ipc:///tmp/spike.tick_timer_connection"

# TCP Connection (For network communication)
TICKER_SOCKET_GLOBAL_NAME = "tcp://*:10000"

class SocketDefinition(object):
    def __init__(self, endpoint, socket_type, is_server = False):
        self.endpoint = endpoint
        self.socket_type = socket_type
        self.is_server = is_server
        self.socket = None

    def create_socket(self, context, name):
        if self.socket is None:
            self.socket = context.socket(self.socket_type)
            print "|||||||||||||||made a socket ",self.endpoint
            if self.is_server:
                self.socket.bind(self.endpoint)
            else:
                self.socket.connect(self.endpoint)
        else:
            print "|||||||||||||||ALREADY EXISTED",self.endpoint

        return self.socket

def create_socket_defs_reqrep(src, dest):
    socket_name = "ipc:///tmp/spike.node_connection.{src_str}-to-{dest_str}".format(
        src_str=src,
        dest_str=dest)

    origin_socket_type = zmq.REQ
    destination_socket_type = zmq.REP

    return SocketDefinition(socket_name, origin_socket_type, is_server=True), SocketDefinition(socket_name, destination_socket_type, is_server=False)

def create_socket_defs_pushpull(src, dest):
    socket_name = "ipc:///tmp/spike.node_connection.{src_str}-to-{dest_str}".format(
        src_str=src,
        dest_str=dest)

    origin_socket_type = zmq.PUSH
    destination_socket_type = zmq.PULL

    return SocketDefinition(socket_name, origin_socket_type, is_server=True), SocketDefinition(socket_name, destination_socket_type, is_server=False)

class Socket(object):
    def __init__(self, definition, name):
        self.definition = definition
        self.name = name

        self.instance = None

    def get_instance(self):
        if self.instance is None:
            print "HOLY FUCK IT WAS NULL************************************************************"
        return self.instance

    def init(self, zmq_context):
        self.instance = self.definition.create_socket(zmq_context,self.name)
        return self.instance

