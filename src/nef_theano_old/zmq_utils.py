import zmq

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

def create_ipc_socket_defs_reqrep(src_name, dest_name):
    return _create_ipc_socket_defs(src_name, dest_name, zmq.REQ, zmq.REP)

def create_ipc_socket_defs_pushpull(src_name, dest_name):
    return _create_ipc_socket_defs(src_name, dest_name, zmq.PUSH, zmq.PULL)

def _create_ipc_socket_defs(src_name, dest_name, src_socket_type, dst_socket_type):
    src_socket_name = dest_socket_name = \
        "ipc:///tmp/spike.node_connection.{src_str}-to-{dest_str}".format(
            src_str=src_name,
            dest_str=dest_name)

    return SocketDefinition(src_socket_name, src_socket_type, is_server=False),\
        SocketDefinition(dest_socket_name, dst_socket_type, is_server=True)

def create_tcp_socket_defs_reqrep(dst_host, port):
    return _create_tcp_socket_defs(dst_host, port, zmq.REQ, zmq.REP)

def create_tcp_socket_defs_pushpull(dst_host, port):
    return _create_tcp_socket_defs(dst_host, port, zmq.PUSH, zmq.PULL)

def _create_tcp_socket_defs(dst_host, port, src_socket_type, dst_socket_type):
    """ Defines two sockets given a dst host name and a port
    """
    dst_addr = "tcp://%s:%s" % (dst_host, port)
    dst_addr_at_dst = "tcp://*:%s" % port

    src = SocketDefinition(dst_addr, src_socket_type, is_server=False)
    dst = SocketDefinition(dst_addr_at_dst, dst_socket_type, is_server=True)
    return src, dst


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

