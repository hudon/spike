import os, getopt
import random
import sys
import uuid
from _collections import OrderedDict

import sys, time

import probe, ensemble, subnetwork, input
import numpy as np
import zmq_utils, zmq

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
        self.local_workers = {}
        self.probe_clients = {}
        self.split_ensembles = {}

        self.job_id = str(uuid.uuid4())
        self.usr_module = usr_module
        self.distributor = distribution.DistributionManager(hosts_file, self.job_id)

        if usr_module is not None:
            sys.stderr.write("DEBUG: Sending user-defined functions file '{0}' to remote hosts.\n".format(self.usr_module))

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


    def _connect_pre_local(self, pre_name, post_addr, func=None):
        node = self.local_workers[pre_name].node
        if not isinstance(node, input.Input):
            raise Exception("ERROR: connect_pre_local only supports Inputs at the moment")
        if func is None:
            origin = node.origin['X']
        else:
            origin_name = func.__name__
            if origin_name not in node.origin:
                node.add_origin(origin_name, func, dt=self.dt, **kwargs)
            origin = node.origin[origin_name]

        # connect origin to post
        socket = zmq_utils.SocketDefinition(post_addr, zmq.PUSH, is_server=False)
        origin.add_output(socket)

        return {
            'pre_output': origin.decoded_output.get_value(),
            'pre_dimensions': origin.dimensions,
            # The connect_pre functions must return these keys even if we know
            # we are not connecting an ensemble
            'pre_array_size': None, 'pre_neurons_out': None, 'pre_neurons_num': None
        }

    def _connect(self, pre, post, func, decoders, **kwargs):
        post_worker = self.workers[post]

        post_port = post_worker.send_command({
            'cmd': 'next_avail_port',
            'args': (),
            'kwargs': {}
        })
        post_addr = 'tcp://%s:%s' % (post_worker.host, post_port)

        if pre in self.local_workers:
            pre_params = self._connect_pre_local(pre, post_addr=post_addr, func=func)
        else:
            pre_params = self.workers[pre].send_command({
                'cmd': 'connect_pre',
                'args': (),
                'kwargs': {'post_addr': post_addr, 'func': func, 'dt': self.dt,
                    'decoders': decoders}
            })

        # Adding pre_params to kwargs
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
                if orig_ensemble.neurons_num % num_subs != 0:
                    raise Exception('ERROR: The number of neurons is not divisible by num_subs')
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
        kwargs['dt'] = self.dt

        inode = input.Input(*args, **kwargs)
        self.local_workers[inode.name] = \
            self.distributor.new_worker(inode.name, True, inode)

    def _make_probe(self, target, name, dt_sample=0.01, data_type='decoded', **kwargs):
        probe_worker = self.distributor.new_worker(name)
        probe_port = probe_worker.send_command({
            'cmd': 'next_avail_port',
            'args': (),
            'kwargs': {}
        })
        probe_addr = 'tcp://%s:%s' % (probe_worker.host, probe_port)

        if target in self.local_workers:
            target_params = self._connect_pre_local(target,
                    post_addr=probe_addr)
        else:
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
        # ensure there are no probes (or probes for subensembles) with the
        # same name
        while name is None or self.workers.has_key(name) or \
                self.workers.has_key(name + '-SUB-0'):
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

    def set_alias(self, alias, name):
        if name in self.local_workers:
            self.workers[alias] = self.local_workers[name]
        else:
            self.workers[alias] = self.workers[name]

    def without_aliases(self, the_dict):
        result = dict()
        for value in the_dict.values():
            if not value.name in result:
                result[value.name] = value
        return result

    def run(self, sim_time):
        start_time = time.time()
        # cleanup data that we do not need anymore
        self.split_ensembles = dict()

        hosts = set([worker.host for worker in self.workers.values()])
        full_addrs = set([worker.daemon_addr for worker in self.workers.values()])

        sys.stderr.write(
            "DEBUG: Starting job {0} on {1} hosts: {2}\n".format(
                self.job_id,
                len(full_addrs),
                "\n".join(full_addrs)
            )
        )

        # Unlock unused hosts (if any)
        for host in set(self.distributor.remote_hosts).difference(hosts):
            self.distributor.unlock(host)

        local_workers = self.without_aliases(self.local_workers)
        workers = self.without_aliases(self.workers)

        for worker in local_workers.values():
            worker.start(sim_time)
        for worker in workers.values():
            worker.send_command({
                'cmd': 'start',
                'args': (worker.worker_port, sim_time),
                'kwargs': {}
            })

        # waiting for a FIN from each worker and sending an ACK for it to finish
        for worker in workers.values():
            worker.send_command({'cmd': 'fin', 'args': (), 'kwargs': {}})

        for worker in local_workers.values():
            worker.send_pyobj("FIN")
        for worker in local_workers.values():
            worker.recv_pyobj() #ack

        for worker in workers.values():
            if worker.name in self.probe_clients.keys():
                client = self.probe_clients[worker.name]
                data = worker.send_command({'cmd': 'get_data', 'args': (), 'kwargs': {}})
                client.add_data(data)
            worker.kill()

        for worker in local_workers.values():
            worker.send_pyobj("KILL")

        self.run_time += sim_time
        sys.stderr.write("run() seconds simulated:" + str(sim_time))
        sys.stderr.write("\nrun() seconds on wall clock:")
        sys.stderr.write(str(time.time() - start_time) + "\n")
