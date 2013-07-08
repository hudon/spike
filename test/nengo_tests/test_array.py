"""This is a file to test the network array function, both with make_array, 
and by using the array_size parameter in the network.make command.

"""

import numpy as np
#import matplotlib.pyplot as plt

import sys
sys.path.append(sys.argv[1])
import nef_theano as nef

def test_array():

    neurons = 40

    is_spike = len(sys.argv) > 2 and sys.argv[2] == 'target'

    net = nef.Network('Array Test', seed=50)
    net.make_input('in', np.arange(-1, 1, .34))
    #net.make_input('in', value=1, zero_after=1.0)

    if is_spike:
        net.make('B', neurons=neurons, array_size=3, dimensions=2,
                is_printing=True)
    else:
        net.make('B', neurons=neurons, array_size=3, dimensions=2)

    net.make_array('A', neurons=neurons, array_size=1, dimensions=6)
    net.make('A2', neurons=neurons, array_size=2, dimensions=3)
    net.make('B2', neurons=neurons, array_size=6, dimensions=1)

    net.connect('in', 'A')
    net.connect('in', 'A2')
    net.connect('in', 'B')
    net.connect('in', 'B2')
    net.connect('A2', 'B')

    timesteps = 0.3
    dt_step = 0.01
    t = np.linspace(dt_step, timesteps*dt_step, timesteps)
    pstc = 0.01


    ## TODO: Spike does not support probes yet
    #Ip = net.make_probe('in', dt_sample=dt_step, pstc=pstc)
    #Ap = net.make_probe('A', dt_sample=dt_step, pstc=pstc)
    #A2p = net.make_probe('A2', dt_sample=dt_step, pstc=pstc)
    #Bp = net.make_probe('B', dt_sample=dt_step, pstc=pstc)
    #B2p = net.make_probe('B2', dt_sample=dt_step, pstc=pstc)

    print "starting simulation"
    if is_spike:
        net.run(timesteps*dt_step)
    else:
        net.run(timesteps * dt_step, print_origin='B')

    # plot the results
    #plt.ioff(); plt.close(); 
    #plt.subplot(5,1,1); plt.ylim([-1.5,1.5])
    #plt.plot(t, Ip.get_data(), 'x'); plt.title('Input')
    #plt.subplot(5,1,2); plt.ylim([-1.5,1.5])
    #plt.plot(Ap.get_data()); plt.title('A, array_size=1, dim=6')
    #plt.subplot(5,1,3); plt.ylim([-1.5,1.5])
    #plt.plot(A2p.get_data()); plt.title('A2, array_size=2, dim=3')
    #plt.subplot(5,1,4); plt.ylim([-1.5,1.5])
    #plt.plot(Bp.get_data()); plt.title('B, array_size=3, dim=2')
    #plt.subplot(5,1,5); plt.ylim([-1.5,1.5])
    #plt.plot(B2p.get_data()); plt.title('B2, array_size=6, dim=1')
    #plt.tight_layout()
    #plt.show()


if __name__ == "__main__":
    test_array()

