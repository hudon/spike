import numpy as np

# types registry (plugin pattern)
# The types registry maps strings like 'lif' and 'lif-rate'
# to sub-classes in lif.py and lif_rate.py
types = {}


def accumulate(J, neurons, dt, time=1.0, init_time=0.05):
    """Accumulates neuron output over time.

    Take a neuron model, run it for the given amount of time with
    fixed input. Used to generate activity matrix when calculating
    origin decoders.
    
    Returns the accumulated output over that time.

    :param Neuron neuron: population of neurons from which to accumulate data
    :param float time: length of time to simulate population for (s)
    :param float init_time: run neurons for this long before collecting data
                            to get rid of startup transients (s)

    """
    ### make the standard neuron update function

    # updates is dictionary of variables returned by neuron.update
    updates = neurons.update(J.astype('float32'), dt)

    # update all internal state variables listed in updates
    
    ### make a variant that also includes computing the total output
    # add another internal variable to change to updates dictionary
    updates[total] = total + neurons.output


    return total.get_value().astype('float32') / time


class Neuron(object):
    """Superclass for neuron models.

    All neurons must implement an update function,
    and should most likely define a more complicated reset function.

    """

    def __init__(self, size):
        """Constructor for neuron model superclass.

        :param int size: number of neurons in this population

        """
        self.size = size

    def reset(self):
        """Reset the state of the neuron."""
        self.output.set_value(np.zeros(self.size).astype('float32'))

    def update(self, input_current):
        """All neuron subclasses must have an update function.

        The update function takes in input_current and returns
        activity information.

        """
        raise NotImplementedError()

