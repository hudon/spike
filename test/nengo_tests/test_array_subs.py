"""This is a file to test the network array function, both with make_array,
and by using the array_size parameter in the network.make command.

"""

import numpy as np
# import matplotlib.pyplot as plt

import sys
sys.path.append(sys.argv[1])
import nef_theano as nef

def test_array():
    neurons = 40

    net = nef.Network('Array Test', seed=51, command_arguments=sys.argv[2:])
    net.make_input('in', np.arange(-1, 1, .34), zero_after_time=1.0)
    net.make('B', neurons=neurons, array_size=3, dimensions=2, num_subs=5)
    net.make_array('A', neurons=neurons, array_size=1, dimensions=6)
    net.make('A2', neurons=neurons, array_size=2, dimensions=3, num_subs=10)
    net.make('B2', neurons=neurons, array_size=6, dimensions=1, num_subs=2)

    net.connect('in', 'A')
    net.connect('in', 'A2')
    net.connect('in', 'B')
    net.connect('in', 'B2')
    net.connect('A2', 'B')

    timesteps = 200
    dt_step = 0.01
    t = np.linspace(dt_step, timesteps*dt_step, timesteps)
    pstc = 0.01

    Ip = net.make_probe('in', dt_sample=dt_step, pstc=pstc)
    Ap = net.make_probe('A', dt_sample=dt_step, pstc=pstc)
    A2p = net.make_probe('A2', dt_sample=dt_step, pstc=pstc)
    Bp = net.make_probe('B', dt_sample=dt_step, pstc=pstc)
    B2p = net.make_probe('B2', dt_sample=dt_step, pstc=pstc)

    print "starting simulation"

    net.run(timesteps * dt_step)

    ip_data = Ip.get_data()
    ap_data = Ap.get_data()
    a2p_data = A2p.get_data()
    bp_data = Bp.get_data()
    b2p_data = B2p.get_data()

    print "input 'in' probe data"
    for x in ip_data:
        print x
    print "ensemble 'A' probe data"
    for x in ap_data:
        print x
    print "ensemble 'A2' probe data"
    for x in a2p_data:
        print x
    print "ensemble 'B' probe data"
    for x in bp_data:
        print x
    print "ensemble 'B2' probe data"
    for x in b2p_data:
        print x

    # plot the results

    # plt.ioff(); plt.close();
    # plt.subplot(5,1,1); plt.ylim([-1.5,1.5])
    # plt.plot(t, ip_data, 'x'); plt.title('Input')
    # plt.subplot(5,1,2); plt.ylim([-1.5,1.5])
    # plt.plot(ap_data); plt.title('A, array_size=1, dim=6')
    # plt.subplot(5,1,3); plt.ylim([-1.5,1.5])
    # plt.plot(a2p_data); plt.title('A2, array_size=2, dim=3')
    # plt.subplot(5,1,4); plt.ylim([-1.5,1.5])
    # plt.plot(bp_data); plt.title('B, array_size=3, dim=2')
    # plt.subplot(5,1,5); plt.ylim([-1.5,1.5])
    # plt.plot(b2p_data); plt.title('B2, array_size=6, dim=1')
    # plt.tight_layout()
    # plt.show()


if __name__ == "__main__":
    test_array()

