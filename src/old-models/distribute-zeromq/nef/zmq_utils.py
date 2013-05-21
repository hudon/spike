import zmq

# Straight-up IPC connection between processes (Does NOT work on Windows)
TICKER_SOCKET_LOCAL_NAME = "ipc:///tmp/spike.tick_timer_connection"

# TCP Connection (For network communication)
TICKER_SOCKET_GLOBAL_NAME = "tcp://*:10000"

class SocketDefinition(object):
    def __init__(self, endpoint, socket_type, is_server=False):
        self.endpoint = endpoint
        self.socket_type = socket_type
        self.is_server = is_server

    def create_socket(self):
        ctx = zmq.Context()
        socket = ctx.socket(self.socket_type)
        if self.is_server:
            socket.bind(self.endpoint)
        else:
            socket.connect(self.endpoint)

        return socket

def create_local_socket_definition_pair(origin_node, destination_node):
    socket_name = ("ipc:///tmp/spike.node_connection"
                  ".{origin_name}-to-{destination_name}").format(
                          origin_name=origin_node.name,
                          destination_name=destination_node.name)

    origin_socket_type = zmq.PUSH
    destination_socket_type = zmq.PULL

    return (SocketDefinition(socket_name, origin_socket_type, is_server=True),
            SocketDefinition(socket_name, destination_socket_type,
                is_server=False))
