import zmq
from multiprocessing import Process
# from threading import Thread

from nef_theano import ensemble, input, origin, probe
import functions

# Distributor listens on this port for commands
LISTENER_ENDPOINT = "tcp://*:10010"

class DistributionDaemon:
    def __init__(self):
        self.zmq_context = zmq.Context()
        self.listener_socket = None
        self.processes = {}

    def spawn_worker(self, node):
        # No arguments. just call run()
        print "Starting worker for node %s" % node.name
        process = Process(target=node.run, name=node.name)
        self.processes[node.name] = process
        process.start()
        print "Worker %s: PID %s" % (node.name, process.pid)

    def listen(self, endpoint):
        self.listener_socket = self.zmq_context.socket(zmq.REP)
        self.listener_socket.bind(endpoint)

        while True:
            msg = self.listener_socket.recv_pyobj()
            print "Received Message: {node}".format(node=node)

            if isinstance(msg, tuple):
                action, name = msg
                if action is FIN:
                    process = self.processes[name]
                    process.join()
            else:
                node = msg
                self.spawn_worker(node)
                self.listener_socket.send("ACK")

def main():
    DistributionDaemon().listen(LISTENER_ENDPOINT)

if __name__ == '__main__':
    main()
