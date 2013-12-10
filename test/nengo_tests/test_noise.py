"""This is a test file to test the noise parameter on ensemble"""

import math

import numpy as np
#import matplotlib.pyplot as plt

import sys, getopt
sys.path.append(sys.argv[1])
import nef_theano as nef

hosts_file = None

optlist, args = getopt.getopt(sys.argv[2:], 's', ['hosts='])
for opt, arg in optlist:
    if opt == '--hosts':
        hosts_file = arg if arg else None

if hosts_file:
  net = nef.Network('Noise Test', seed=91, hosts_file=hosts_file)
else:
  net = nef.Network('Noise Test', seed=91)

net.make_input('in', value=math.sin)
net.make('A', neurons=300, dimensions=1, noise=1)
net.make('A2', neurons=300, dimensions=1, noise=100)
net.make('B', neurons=300, dimensions=2, noise=1000, noise_type='Gaussian')
net.make('C', neurons=100, dimensions=1, array_size=3, noise=10)

net.connect('in', 'A')
net.connect('in', 'A2')
net.connect('in', 'B')
net.connect('in', 'C')

timesteps = 500
dt_step = 0.01
t = np.linspace(dt_step, timesteps*dt_step, timesteps)
pstc = 0.01

Ip = net.make_probe('in', dt_sample=dt_step, pstc=pstc)
Ap = net.make_probe('A', dt_sample=dt_step, pstc=pstc)
A2p = net.make_probe('A2', dt_sample=dt_step, pstc=pstc)
Bp = net.make_probe('B', dt_sample=dt_step, pstc=pstc)
Cp = net.make_probe('C', dt_sample=dt_step, pstc=pstc)

print "starting simulation"
net.run(timesteps * dt_step)

ip_data = Ip.get_data()
ap_data = Ap.get_data()
a2p_data = A2p.get_data()
bp_data = Bp.get_data()
cp_data = Cp.get_data()


print "input 'in' probe data"
for x in ip_data:
    print x

print "input 'ap' probe data"
for x in ap_data:
    print x

print "input 'a2p' probe data"
for x in a2p_data:
    print x

print "input 'bp' probe data"
for x in bp_data:
    print x

print "input 'cp' probe data"
for x in cp_data:
    print x

# plot the results
#plt.ioff(); plt.close()
#plt.subplot(411); plt.title('Input')
#plt.plot(Ip.get_data())
#plt.subplot(412); plt.hold(1)
#plt.plot(Ap.get_data()); plt.plot(A2p.get_data())
#plt.legend(['A noise = 1', 'A2 noise = 100'])
#plt.subplot(413); plt.title('B noise = 1000, type = gaussian')
#plt.plot(Bp.get_data())
#plt.subplot(414); plt.title('C')
#plt.plot(Cp.get_data())
#plt.tight_layout()
#plt.show()
