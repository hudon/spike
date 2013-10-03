from _collections import OrderedDict

import numpy as np

from . import neuron
from . import cache
from .origin import Origin

class EnsembleOrigin(Origin):
    def __init__(self, ensemble, dt, func=None, eval_points=None):
        """The output from a population of neurons (ensemble),
        performing a transformation (func) on the represented value.

        :param Ensemble ensemble:
            the Ensemble to which this origin is attached
        :param function func:
            the transformation to perform to the ensemble's
            represented values to get the output value

        """
        self.ensemble = ensemble
        # sets up self.decoders
        func_size = 1#self.compute_decoders(func, dt, eval_points) 
        # decoders is array_size * neurons_num * func_dimensions, 
        # initial value should have array_size values * func_dimensions
        initial_value = np.zeros(self.ensemble.array_size * func_size) 
        Origin.__init__(self, func=func, initial_value=initial_value)
        self.func_size = func_size

    def update(self, dt, spikes):
        """the computation for converting neuron output
        into a decoded value.

        returns a dictionary with the decoded output value

        :param array spikes:
            object representing the instantaneous spike raster
            from the attached population

        """

        # multiply the output by the attached ensemble's radius
        # to put us back in the right range
        r = self.ensemble.radius
        # weighted summation over neural activity to get decoded_output
        z = TT.zeros((self.ensemble.array_size, self.func_size), dtype='float32')

        return OrderedDict({})
