import os, getopt
import random
from _collections import OrderedDict

import probe, ensemble, subnetwork
import numpy as np

from . import distribution

class Network(object):
    def __init__(self, name, command_arguments=None, seed=None, fixed_seed=None, dt=.001,
        usr_module=None):
        hosts_file = None

        optlist, args = getopt.getopt(command_arguments, 's', ['hosts='])
        for opt, arg in optlist:
            if opt == '--hosts':
                hosts_file = arg if arg else None

        if hosts_file is None:
            raise Exception("ERROR: A hosts_file is required (try the --hosts arg)")

        self.workers = {}
        self.probe_clients = {}
        self.split_ensembles = {}
        self.distributor = distribution.DistributionManager(hosts_file)
        if usr_module is not None:
            module_name = os.path.basename(usr_module)
            with open(usr_module, "rb") as module:
                self.distributor.send_usr_module(module_name, module.read())

        self.name = name
        self.dt = dt
        self.run_time = 0.0
        self.seed = seed
        np.random.seed(seed)
        self.fixed_seed = fixed_seed

        self.random = random.Random()
        if seed is not None:
            self.random.seed(seed)

    def _connect(self, pre, post, func, decoders, **kwargs):
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
            'kwargs': {'post_addr': post_addr, 'func': func, 'dt': self.dt,
                'decoders': decoders}
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

    def connect(self, pre, post, func=None, **kwargs):
        if pre in self.split_ensembles:
            is_pre_split = True
            pres = self.split_ensembles[pre]['children']

            origin_name = func.__name__ if func is not None else 'X'
            decoders = self.split_ensembles[pre]['parent'].get_subensemble_decoder(
                len(pres), origin_name, func)
        else:
            is_pre_split = False
            pres = [pre]

        if post in self.split_ensembles:
            posts = self.split_ensembles[post]['children']
        else:
            posts = [post]

        for i, pre in enumerate(pres):
            decoder = decoders[i] if is_pre_split else None
            for post in posts:
                self._connect(pre, post, func, decoder, **kwargs)

    def _make_ensemble(self, name, **kwargs):
        worker = self.distributor.new_worker(name)
        worker.send_command({
            'cmd': 'make_ensemble',
            'args': (name, ),
            'kwargs': kwargs
        })
        self.workers[name] = worker

    def make(self, name, num_subs=1, **kwargs):
        if num_subs < 1:
            raise Exception("ERROR", "num_subs must be greater than 0")

        if 'seed' not in kwargs.keys():
            if self.fixed_seed is not None:
                kwargs['seed'] = self.fixed_seed
            else:
                kwargs['seed'] = self.random.randrange(0x7fffffff)
        kwargs['dt'] = self.dt

        if num_subs == 1:
            self._make_ensemble(name, **kwargs)
        else:
            if 'mode' in kwargs and kwargs['mode'] == 'direct':
                raise Exception("ERROR", "do not support direct subensembles")

            orig_ensemble = ensemble.Ensemble(**kwargs)
            self.split_ensembles[name] = {
                'parent': orig_ensemble,
                'children': []
            }
            e_num = 0
            for encoder, decoder, bias, alpha in \
                orig_ensemble.get_subensemble_parts(num_subs):

                sub_name = name + "-SUB-" + str(e_num)
                e_num += 1

                kwargs["dimensions"] = orig_ensemble.dimensions
                kwargs["neurons"] = orig_ensemble.neurons_num / num_subs
                kwargs["encoders"] = encoder
                kwargs["decoders"] = decoder
                kwargs["bias"] = bias
                kwargs["alpha"] = alpha
                self._make_ensemble(sub_name, is_subensemble=True, **kwargs)
                self.split_ensembles[name]['children'].append(sub_name)

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

    def _make_probe(self, target, name, dt_sample=0.01, data_type='decoded', **kwargs):
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
        kwargs['target'] = target_params['pre_output'] # target origin output

        probe_worker.send_command({
            'cmd': 'make_probe',
            'args': (probe_port, name), # necessary to pass name directly to probe constructor
            'kwargs': kwargs
        })
        self.workers[name] = probe_worker

    def make_probe(self, target, name=None, dt_sample=0.01, data_type='decoded', **kwargs):
        i = 0
        while name is None or self.workers.has_key(name):
            i += 1
            name = ("Probe%d" % i)

        # a wrapper for probe used by users to retrieve probe data
        client = probe.ProbeClient(name)

        if target in self.split_ensembles:
            target_subs = self.split_ensembles[target]['children']
            for i, target_sub in enumerate(target_subs):
                sub_name = name + "-SUB-" + str(i)
                self._make_probe(target_sub, sub_name, dt_sample, data_type, **kwargs)
                self.probe_clients[sub_name] = client
        else:
            self._make_probe(target, name, dt_sample, data_type, **kwargs)
            self.probe_clients[name] = client

        return client

    def make_subnetwork(self, name):
        return subnetwork.SubNetwork(name, self)

    def run(self, time):
        # cleanup data that we do not need anymore
        self.split_ensembles = dict()

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
                client.add_data(data)
            worker.kill()

        self.run_time += time
