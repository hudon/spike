"""This is a file to test the encoders parameter on ensembles"""

import math
import time

import numpy as np
#import matplotlib.pyplot as plt

import sys, getopt
sys.path.append(sys.argv[1])
import nef_theano as nef

import functions

build_time_start = time.time()

timesteps = 1000
dt_step = 0.001

net = nef.Network('Encoder Test', dt=dt_step, seed=103, command_arguments=sys.argv[2:], usr_module='test/nengo_tests/functions.py')

net.make_input('in1', math.sin)
net.make_input('in2', math.cos)
net.make('A', neurons=100, dimensions=1)
net.make('B', neurons=100, dimensions=1, encoders=[[1]], intercept=(0, 1.0))
net.make('C', neurons=100, dimensions=2, radius=1.5)
net.make('D', neurons=100, dimensions=2, encoders=[[1,1],[1,-1],[-1,-1],[-1,1]], radius=1.5)
net.make('outputC', neurons=1, dimensions=1, mode='direct')
net.make('outputD', neurons=1, dimensions=1, mode='direct')

net.connect('in1', 'A')
net.connect('A', 'B')
net.connect('in1', 'C', index_post=[0])
net.connect('in2', 'C', index_post=[1])
net.connect('in1', 'D', index_post=[0])
net.connect('in2', 'D', index_post=[1])

net.connect('C', 'outputC', func=functions.product)
net.connect('D', 'outputD', func=functions.product)

t = np.linspace(dt_step, timesteps*dt_step, timesteps)
pstc = 0.01
#I1p = net.make_probe('in1', dt_sample=dt_step, pstc=pstc)
#I2p = net.make_probe('in2', dt_sample=dt_step, pstc=pstc)
#Ap = net.make_probe('A', dt_sample=dt_step, pstc=pstc)
#Bp = net.make_probe('B', dt_sample=dt_step, pstc=pstc)
#Cp = net.make_probe('outputC', dt_sample=dt_step, pstc=pstc)
#Dp = net.make_probe('outputD', dt_sample=dt_step, pstc=pstc)

build_time_end = time.time()

print "starting simulation"
net.run(timesteps * dt_step)

sim_time_end = time.time()
#print "\nBuild time: %0.10fs" % (build_time_end - build_time_start)
#print "Sim time: %0.10fs" % (sim_time_end - build_time_end)

#plt.ioff(); plt.close(); plt.hold(1)
#plt.subplot(211)
#plt.plot(t, I1p.get_data())
#plt.plot(t, Ap.get_data())
#plt.plot(t, Bp.get_data())
#plt.legend(['Input', 'A', 'B'])
#
#plt.subplot(212); plt.title('Multiplication test')
#plt.plot(t, I1p.get_data() * I2p.get_data())
#plt.plot(t, Cp.get_data())
#plt.plot(t, Dp.get_data())
#plt.legend(['Answer', 'C', 'D'])
#plt.tight_layout()
#plt.show()

