"""This is a test file to test the func parameter on the connect method"""

import math

import numpy as np
#import matplotlib.pyplot as plt

import sys
sys.path.append(sys.argv[1])
import nef_theano as nef

import functions as funcs



net = nef.Network('Function Test', seed=91, command_arguments=sys.argv[2:],
  usr_module='test/nengo_tests/functions.py')
net.make_input('in', value=math.sin)
net.make('A', neurons=250, dimensions=1)
net.make('B', neurons=250, dimensions=3)

net.connect('in', 'A')
net.connect('A', 'B', func=funcs.square, pstc=0.1)

timesteps = 500
dt_step = 0.01
t = np.linspace(dt_step, timesteps*dt_step, timesteps)
pstc = 0.03

Ip = net.make_probe('in', dt_sample=dt_step, pstc=pstc)
Ap = net.make_probe('A', dt_sample=dt_step, pstc=pstc)
Bp = net.make_probe('B', dt_sample=dt_step, pstc=pstc)

print "starting simulation"
net.run(timesteps * dt_step)

ip_data = Ip.get_data()
ap_data = Ap.get_data()
bp_data = Bp.get_data()

print "input 'in' probe data"
for x in ip_data:
    print x
print "ensemble 'A' probe data"
for x in ap_data:
    print x
print "ensemble 'B' probe data"
for x in bp_data:
    print x
# plot the results
#plt.ioff(); plt.clf(); plt.hold(1);
#plt.plot(Ip.get_data())
#plt.plot(Ap.get_data())
#plt.plot(Bp.get_data())
#plt.legend(['Input','A','B0','B1','B2'])
#plt.tight_layout()
#plt.show()
