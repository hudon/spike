import os
import random
from _collections import OrderedDict

import probe

from . import distribution

class Network(object):
    def __init__(self, name, hosts_file, seed=None, fixed_seed=None, dt=.001,
        usr_module=None):

        self.workers = {}
        self.probe_clients = {}
        self.distributor = distribution.DistributionManager(hosts_file)
        if usr_module is not None:
            module_name = os.path.basename(usr_module)
            with open(usr_module, "rb") as module:
                self.distributor.send_usr_module(module_name, module.read())

        self.name = name
        self.dt = dt
        self.run_time = 0.0
        self.seed = seed
        self.fixed_seed = fixed_seed

        self.random = random.Random()
        if seed is not None:
            self.random.seed(seed)

    def connect(self, pre, post, func=None, **kwargs):
        pre_worker = self.workers[pre]
        post_worker = self.workers[post]

        post_port = post_worker.send_command({
            'cmd': 'next_avail_port',
            'args': (),
            'kwargs': {}
        })
        post_addr = 'tcp://%s:%s' % (post_worker.host, post_port)

        pre_params = pre_worker.send_command({
            'cmd': 'connect_pre',
            'args': (),
            'kwargs': {'post_addr': post_addr, 'func': func, 'dt': self.dt}
        })

        for key, value in pre_params.iteritems():
            kwargs[key] = value
        kwargs['pre_name'] = pre
        kwargs['post_port'] = post_port
        post_worker.send_command({
            'cmd': 'connect_post',
            'args': (),
            'kwargs': kwargs
        })

    def make(self, name, *args, **kwargs):
        worker = self.distributor.new_worker(name)

        if 'seed' not in kwargs.keys():
            if self.fixed_seed is not None:
                kwargs['seed'] = self.fixed_seed
            else:
                kwargs['seed'] = self.random.randrange(0x7fffffff)
        kwargs['dt'] = self.dt
        args = (name, ) + args
        worker.send_command({
            'cmd': 'make_ensemble',
            'args': args,
            'kwargs': kwargs
        })
        self.workers[name] = worker

    def make_array(self, name, neurons, array_size, dimensions=1, **kwargs):
        self.make(name=name, neurons=neurons, dimensions=dimensions,
            array_size=array_size, **kwargs)

    def make_input(self, *args, **kwargs):
        worker = self.distributor.new_worker(args[0])

        kwargs['dt'] = self.dt
        worker.send_command({
            'cmd': 'make_input',
            'args': args,
            'kwargs': kwargs
        })
        self.workers[args[0]] = worker

    def make_probe(self, target, name=None, dt_sample=0.01, data_type='decoded', **kwargs):
        i = 0
        while name is None or self.workers.has_key(name):
            i += 1
            name = ("Probe%d" % i)

        probe_worker = self.distributor.new_worker(name)
        probe_port = probe_worker.send_command({
            'cmd': 'next_avail_port',
            'args': (),
            'kwargs': {}
        })
        probe_addr = 'tcp://%s:%s' % (probe_worker.host, probe_port)

        target_worker = self.workers[target] # TODO: handle origin targets case
        target_params = target_worker.send_command({
            'cmd': 'connect_pre',
            'args': (),
            'kwargs': {'post_addr': probe_addr}
        })

        kwargs['dt'] = self.dt
        kwargs['dt_sample'] = dt_sample
        kwargs['target_name'] = target + '-' + data_type
        kwargs['target'] = target_params['pre_output']

        probe_worker.send_command({
            'cmd': 'make_probe',
            'args': (probe_port, name), # necessary to pass name directly to probe constructor
            'kwargs': kwargs
        })
        self.workers[name] = probe_worker

        # a wrapper for probe used by users to retrieve probe data
        client = probe.ProbeClient(name)
        self.probe_clients[name] = client
        return client

    def run(self, time):
        for worker in self.workers.values():
            worker.send_command({
                'cmd': 'start',
                'args': (worker.worker_port, time),
                'kwargs': {}
            })

        # waiting for a FIN from each worker and sending an ACK for it to finish
        for worker in self.workers.values():
            worker.send_command({'cmd': 'fin', 'args': (), 'kwargs': {}})

        for worker in self.workers.values():
            if worker.name in self.probe_clients.keys():
                client = self.probe_clients[worker.name]
                data = worker.send_command({'cmd': 'get_data', 'args': (), 'kwargs': {}})
                client.set_data(data)
            worker.kill()

        self.run_time += time
