import zmq
import zmq_utils

from multiprocessing import Process

class Worker:
  def __init__(self, zmq_context, node, is_distributed, host):
    self.node = node
    self.zmq_context = zmq_context
    self.is_distributed = is_distributed

    if is_distributed:
      self.host = host
      self.admin_socket_def, self.node_socket_def = \
        zmq_utils.create_tcp_socket_defs_reqrep("admin", node.name)

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

    if is_distributed:
      socket = self.zmq_context.socket(zmq.REQ)
      socket.connect(self.host)
      socket.send_pyobj({
        "node": self.node,
        "socket": self.node_socket_def
      })
      socket.recv() # wait for an ACK from the daemon
      socket.close()
    else:
      self.process.start()

  def stop(self):
    if is_distributed:
      socket = self.zmq_context.socket(zmq.REQ)
      socket.connect(self.host)
      socket.send_pyobj(('FIN', self.node.name))
      socket.close()
    else:
      self.process.join()


class DistributionManager:
  """ Class responsible for socket creation and work distribution """

  def __init__(self, is_distributed=False):
    self.workers = {}
    self.host = "tcp://localhost:10010";

    self.zmq_context = zmq.Context()
    self.is_distributed = is_distributed

  def create_worker(self, node):
    worker = Worker(self.zmq_context, node, self.is_distributed, self.host)
    self.workers[node.name] = worker
    return worker

  def connect(self, src_name, dst_name):
    """ Will connect 2 workers together
        return: (origin_socket, dest_socket)
    """
    return zmq_utils.create_ipc_socket_defs_pushpull(src_name, dst_name)
