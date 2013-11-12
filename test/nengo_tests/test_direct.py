"""This is a file to test the direct mode on ensembles"""

import math
import time

import numpy as np
#import matplotlib.pyplot as plt

import sys
sys.path.append(sys.argv[1])
import nef_theano as nef

import functions

build_time_start = time.time()

net = nef.Network('Direct Mode Test', seed=47)
net.make_input('in', math.sin)
net.make('A', 100, 1)
net.make('B', 1, 1, mode='direct')
net.make('C', 100, 1)
net.make('D', 1, 2, mode='direct')
net.make('E', 1, array_size=2, dimensions=2, mode='direct')

net.connect('in', 'A')
net.connect('A', 'B')
net.connect('B', 'C')
net.connect('B', 'E')
net.connect('E', 'D', func=functions.product)

timesteps = 1000
dt_step = 0.0001

t = np.linspace(dt_step, timesteps*dt_step, timesteps)
pstc = 0.01
Ip = net.make_probe('in', dt_sample=dt_step, pstc=pstc)
Ap = net.make_probe('A', dt_sample=dt_step, pstc=pstc)
Bp = net.make_probe('B', dt_sample=dt_step, pstc=pstc)
Cp = net.make_probe('C', dt_sample=dt_step, pstc=pstc)
Dp = net.make_probe('D', dt_sample=dt_step, pstc=pstc)
Ep = net.make_probe('E', dt_sample=dt_step, pstc=pstc)



build_time_end = time.time()

print "starting simulation"
net.run(timesteps * dt_step)

sim_time_end = time.time()

Ip_data = Ip.get_data()
Ap_data = Ap.get_data()
Bp_data = Bp.get_data()
Cp_data = Cp.get_data()
Dp_data = Dp.get_data()
Ep_data = Ep.get_data()

print "input 'Ip' probe data"
for x in Ip_data:
    print x
print "ensemble 'Ap' probe data"
for x in Ap_data:
    print x
print "ensemble 'Bp' probe data"
for x in Bp_data:
    print x
print "ensemble 'Cp' probe data"
for x in Cp_data:
    print x
print "ensemble 'Dp' probe data"
for x in Dp_data:
    print x
print "ensemble 'Ep' probe data"
for x in Ep_data:
    print x

#plt.ioff(); plt.close()
#plt.subplot(611); plt.title('Input')
#plt.plot(t, Ip.get_data())
#plt.subplot(612); plt.title('A = spiking')
#plt.plot(t, Ap.get_data())
#plt.subplot(613); plt.title('B = direct')
#plt.plot(t, Bp.get_data())
#plt.subplot(614); plt.title('C = direct')
#plt.plot(t, Cp.get_data())
#plt.subplot(615); plt.title('D = direct')
#plt.plot(t, Dp.get_data())
#plt.subplot(616); plt.title('E = direct')
#plt.plot(t, Ep.get_data())
#plt.tight_layout()
#plt.show()

