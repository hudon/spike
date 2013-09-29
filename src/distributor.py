import zmq
from multiprocessing import Process

from nef_theano import ensemble, input, origin, probe
import functions

# Distributor listens on this port for commands
LISTENER_ENDPOINT = "tcp://*:10010"

class Distributor:
    def __init__(self):
        self.zmq_context = zmq.Context()
        self.listener_socket = None
        self.spawned_workers = []

    def daemonize_worker(self, worker):
        # No arguments. just call run()
        print "Starting worker %s" % worker.name
        worker_process = Process(target=worker.run)
        self.spawned_workers.append(worker_process)
        worker_process.start()
        print "Worker %s: PID %s" % (worker.name, worker_process.pid)

    def listen(self, endpoint):
        self.listener_socket = self.zmq_context.socket(zmq.REP)
        self.listener_socket.bind(endpoint)
        
        while True:
            node = self.listener_socket.recv_pyobj() #Receive a definition for a worker
            print "Received Message: {node}".format(node=node)
            self.daemonize_worker(node)
            self.listener_socket.send("") #Return status of started worker

def main():
    Distributor().listen(LISTENER_ENDPOINT)

if __name__ == '__main__':
    main()
