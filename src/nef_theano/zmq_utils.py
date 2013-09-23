import zmq
import socket
import shlex 
import fcntl

# Straight-up IPC connection between processes (Does NOT work on Windows)
TICKER_SOCKET_LOCAL_NAME = "ipc:///tmp/spike.tick_timer_connection"

# TCP Connection (For network communication)
TICKER_SOCKET_GLOBAL_NAME = "tcp://*:10000"

class SocketDefinition(object):
    def __init__(self, endpoint, socket_type, is_server = False):
        self.endpoint = endpoint
        self.socket_type = socket_type
        self.is_server = is_server

    def create_socket(self, context):
        socket = context.socket(self.socket_type)
        if self.is_server:
            socket.bind(self.endpoint)
        else:
            socket.connect(self.endpoint)

        return socket

def get_ip_address(ifname="eth0"):                             
    """ Get the IP address of a network interface """
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    return socket.inet_ntoa(fcntl.ioctl(                
        s.fileno(),                                     
        0x8915,  # SIOCGIFADDR                          
        struct.pack('256s', ifname[:15])                
    )[20:24])                                           

def create_ipc_socket_defs_reqrep(src_name, dest_name):
    return _create_ipc_socket_defs(src_name, dest_name, zmq.REQ, zmq.REP)

def create_ipc_socket_defs_pushpull(src_name, dest_name):
    return _create_ipc_socket_defs(src_host, dest_host, zmq.PUSH, zmq.PULL)

def _create_ipc_socket_defs(src_name, dest_name, src_socket_type, dest_socket_type):
    origin_socket_type = src_socket_type
    destination_socket_type = dest_socket_type
    origin_socket_name = destination_socket_name = \
    "ipc:///tmp/spike.node_connection.{src_str}-to-{dest_str}".format(
        src_str=src_name,
        dest_str=dest_name)

    return SocketDefinition(origin_socket_name, origin_socket_type, is_server=False), SocketDefinition(destination_socket_name, destination_socket_type, is_server=True)

def create_tcp_socket_defs_reqrep(src_host, dest_host):
    """ Defines two sockets given a pair of TCP 'host:port' hostnames """
    return _create_tcp_socket_defs(src_host, dest_host, zmq.REQ, zmq.REP)

def create_tcp_socket_defs_pushpull(src_host, dest_host):
    return _create_tcp_socket_defs(src_host, dest_host, zmq.PUSH, zmq.PULL)

def _create_tcp_socket_defs(src_host, dest_host, src_socket_type, dest_socket_type):
    origin_socket_type = src_socket_type
    destination_socket_type = dest_socket_type
    origin_socket_name = None
    destination_socket_name = None

    origin_socket_name = "tcp://%s" % src_host
    destination_socket_name = "tcp://%s" % dest_host

    return SocketDefinition(origin_socket_name, origin_socket_type, is_server=False), SocketDefinition(destination_socket_name, destination_socket_type, is_server=True)

class Socket(object):
    def __init__(self, definition, name):
        self.definition = definition
        self.name = name
        self.instance = None

    def get_instance(self):
        return self.instance

    def init(self, zmq_context):
        self.instance = self.definition.create_socket(zmq_context)
        return self.instance

