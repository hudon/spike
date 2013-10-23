import zmq
import zmq_utils

from multiprocessing import Process

class Worker:
  def __init__(self, zmq_context, node, is_distributed):
    self.zmq_context = zmq_context

    self.admin_socket_def, node_socket_def = \
      zmq_utils.create_socket_defs_reqrep("admin", node.name)

    if is_distributed:
      pass
    else:
      self.process = Process(target=node.run, args=(node_socket_def,), name=node.name)

    self.is_distributed = is_distributed

  def send(self, content):
    return self.admin_socket.send(content)

  def recv(self):
    return self.admin_socket.recv()

  def recv_pyobj(self):
    return self.admin_socket.recv_pyobj()

  def start(self):
    self.admin_socket = self.admin_socket_def.create_socket(self.zmq_context)
    if not self.is_distributed:
      self.process.start()

  def stop(self):
    if not self.is_distributed:
      self.process.join()


class DistributionManager:
  """ Class responsible for socket creation and work distribution """

  def __init__(self, is_distributed=False):
    self.workers = {}

    self.zmq_context = zmq.Context()
    self.is_distributed = is_distributed

  def create_worker(self, node):
    worker = Worker(self.zmq_context, node, self.is_distributed)
    self.workers[node.name] = worker
    return worker

  def connect(self, src_name, dst_name):
    """ Will connect 2 workers together
        return: (origin_socket, dest_socket)
    """
    return zmq_utils.create_socket_defs_pushpull(src_name, dst_name)





