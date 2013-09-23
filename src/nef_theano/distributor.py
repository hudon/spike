import zmq
from multiprocessing import Process

# Distributor listens on this port for commands
LISTENER_ENDPOINT = "tcp://*:10010"

class Distributor:
    def __init__(self):
        self.zmq_context = zmq.Context()
        self.listener_socket = None
        self.spawned_workers = []

    def daemonize_worker(self, worker):
        # No arguments. just call .run()
        worker = Process(target=worker.run, daemon=True)
        self.spawned_workers.append(worker)

        worker.start()

        return worker.pid

    def listen(self, endpoint):
        self.listener_socket = self.zmq_context.socket(zmq.REP)
        self.listener_socket.bind(endpoint)
        
        while True:
            node = self.listener_socket.recv_pyobj() #Receive a definition for a worker
            pid = daemonize_worker(node)
            self.listener_socket.send(pid) #Return PID of started worker

def main():
    Distributor().listen(LISTENER_ENDPOINT)

if __name__ == '__main__':
    main()
