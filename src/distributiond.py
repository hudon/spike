import zmq
from multiprocessing import Process
# from threading import Thread

from nef_theano import ensemble, input, origin, probe
import marshal, types

# Distributor listens on this port for commands
LISTENER_ENDPOINT = "tcp://*:9000"

class DistributionDaemon:
    def __init__(self):
        self.zmq_context = zmq.Context()
        self.listener_socket = None
        self.processes = {}

    def spawn_worker(self, node, socket_def):
        # No arguments. just call run()
        print "DEBUG: Starting worker for node %s" % node.name
        process = Process(target=node.run, args=(socket_def,), name=node.name)
        self.processes[node.name] = process
        process.start()
        print "DEBUG: Worker %s: PID %s" % (node.name, process.pid)

    def listen(self, endpoint):
        self.listener_socket = self.zmq_context.socket(zmq.REP)
        self.listener_socket.bind(endpoint)

        while True:
            msg = self.listener_socket.recv_pyobj()
            print "DEBUG: Received Message: {msg}".format(msg=msg)

            if isinstance(msg, tuple):
                action, name = msg
                if action == 'FIN':
                    print "DEBUG: Terminating process:", name
                    process = self.processes[name]
                    process.join()
            else:
                node = msg["node"]
                socket_def = msg["socket"]

                # demarshal origin functions
                if hasattr(node, 'origin'):
                    for o in node.origin.values():
                        if o.func is not None:
                            code = marshal.loads(o.func)
                            o.func = types.FunctionType(code, globals(), 'func')

                self.spawn_worker(node, socket_def)

            self.listener_socket.send("ACK")

def main():
    DistributionDaemon().listen(LISTENER_ENDPOINT)

if __name__ == '__main__':
    main()
