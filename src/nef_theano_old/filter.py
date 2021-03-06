from _collections import OrderedDict

import numpy as np
import theano
import theano.tensor as TT

class Filter:
    """Filter an arbitrary theano.shared"""

    def __init__(self, pstc, name=None, source=None, shape=None, 
        pre_output=None):
        """
        :param float pstc:
        :param string name:
        :param source:
        :type source:
        :param tuple shape:
        """
        self.pstc = pstc
        self.source = source

        # has to be a Theano shared variable
        self.pre_output = pre_output

        ### try to find the shape from the given parameters (source and shape)
        if source is not None and hasattr(source, 'get_value'):
            value = source.get_value()
            if shape is not None:
                assert value.shape == shape
            if name is None: 
                name = 'filtered_%s' % source.name
        elif shape is not None:
            value = np.zeros(shape, dtype='float32')
        else:
            raise Exception("Either \"source\" or \"shape\" must define filter shape")
            
        ### create shared variable to store the filtered value
        self.value = theano.shared(value, name=name)
        self.name = name

    # TODO: will set_value always work in this case?
    def set_pre_output(self, output):
        self.pre_output.set_value(output)

    def set_source(self, source):
        """Set the source of data for this filter

        :param source:
        :type source:
        """
        self.source = source

    def update(self, dt):
        """
        :param float dt: the timestep of the update
        """
        if self.pstc >= dt:
            decay = TT.cast(np.exp(-dt / self.pstc), self.value.dtype)
            value_new = decay * self.value + (1 - decay) * self.source
            return OrderedDict([(self.value, value_new.astype('float32'))])
        else:
            ### no filtering, so just make the value the source
            return OrderedDict([(self.value, self.source.astype('float32'))])
