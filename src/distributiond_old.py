import zmq
from multiprocessing import Process
# from threading import Thread

from nef_theano import ensemble, input, origin, probe
import tempfile
import functions

# Distributor listens on this port for commands
LISTENER_ENDPOINT = "tcp://*:9000"

class DistributionDaemon:
    def __init__(self):
        self.zmq_context = zmq.Context()
        self.listener_socket = None
        self.worker_port = 8000
        self.current_port_offset = 0
        self.processes = {}

    def spawn_worker(self, node, socket_def):
        # No arguments. just call run()
        print "DEBUG: Starting worker for node %s" % node.name
        process = Process(target=node.run, args=(socket_def,), name=node.name)
        self.processes[node.name] = process
        process.start()
        print "DEBUG: Worker %s: PID %s" % (node.name, process.pid)

    def _get_next_port(self):
        next_port = self.worker_port + self.current_port_offset
        self.current_port_offset += 1

        # Roll over after 1000 port assignments
        if self.current_port_offset >= 1000:
            self.current_port_offset = 0

        return next_port

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
            elif msg == 'NEXT_AVAIL_PORT':
                    next_port = self._get_next_port()
                    print "DEBUG: Next available port: {port}".format(port=next_port)
                    self.listener_socket.send(str(next_port))
                    continue
            elif msg == 'PING':
                self.listener_socket.send("PONG")
                continue
            else:
                node = msg["node"]
                socket_def = msg["socket"]

                #NOTE: now we write the functions file and import it so that
                # arbitrary functions may be received
                func_content = msg["functions"]
                with tempfile.NamedTemporaryFile() as tmp:
                    tmp.write(func_content)
                    tmp.flush()
                    execfile(tmp.name)
                self.spawn_worker(node, socket_def)

            self.listener_socket.send("ACK")

def main():
    DistributionDaemon().listen(LISTENER_ENDPOINT)

if __name__ == '__main__':
    main()
