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

def _create_ipc_socket_defs(src_name, dest_name, src_socket_type, dest_socket_type):
    src_socket_name = dest_socket_name = \
        "ipc:///tmp/spike.node_connection.{src_str}-to-{dest_str}".format(
            src_str=src_name,
            dest_str=dest_name)

    return SocketDefinition(src_socket_name, src_socket_type, is_server=True),\
        SocketDefinition(dest_socket_name, dest_socket_type, is_server=False)

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

