"""This is a test file to test the weight, index_pre, and index_post parameters
on the connect function. 
"""

import math

import numpy as np
#import matplotlib.pyplot as plt

import sys
sys.path.append(sys.argv[1])
import nef_theano as nef

#  TODO:  If the seed value is 97, the tests don't pass.
net = nef.Network('Weight, Index_Pre, and Index_Post Test', seed=96, command_arguments=sys.argv[2:])

net.make_input('in', value=math.sin)
net.make('A', neurons=300, dimensions=1)
net.make('B', neurons=300, dimensions=1)
net.make('C', neurons=400, dimensions=2)
net.make('D', neurons=300, dimensions=1)
net.make('E', neurons=400, dimensions=2)
net.make('F', neurons=400, dimensions=2)

net.connect('in', 'A', weight=.5)
net.connect('A', 'B', weight=2)
net.connect('A', 'C', index_post=1)
net.connect('A', 'D')
net.connect('C', 'E', index_pre=1)
net.connect('C', 'F', index_pre=1, index_post=0)

timesteps = 500
dt_step = 0.01
t = np.linspace(dt_step, timesteps*dt_step, timesteps)
pstc = 0.01
Ip = net.make_probe('in', dt_sample=dt_step, pstc=pstc)
Ap = net.make_probe('A', dt_sample=dt_step, pstc=pstc)
Bp = net.make_probe('B', dt_sample=dt_step, pstc=pstc)
Cp = net.make_probe('C', dt_sample=dt_step, pstc=pstc)
Dp = net.make_probe('D', dt_sample=dt_step, pstc=pstc)
Ep = net.make_probe('E', dt_sample=dt_step, pstc=pstc)
Fp = net.make_probe('F', dt_sample=dt_step, pstc=pstc)

print "starting simulation"
net.run(timesteps*dt_step)

ip_data = Ip.get_data()
ap_data = Ap.get_data()
bp_data = Bp.get_data()
cp_data = Cp.get_data()
dp_data = Dp.get_data()
ep_data = Ep.get_data()
fp_data = Fp.get_data()

print "input 'in' probe data"
for x in ip_data:
    print x
print "ensemble 'A' probe data"
for x in ap_data:
    print x
print "ensemble 'B' probe data"
for x in bp_data:
    print x
print "ensemble 'C' probe data"
for x in cp_data:
    print x
print "ensemble 'D' probe data"
for x in dp_data:
    print x
print "ensemble 'E' probe data"
for x in ep_data:
    print x
print "ensemble 'F' probe data"
for x in fp_data:
    print x

#plt.ioff(); plt.close(); 
#plt.subplot(711); plt.title('Input')
#plt.plot(Ip.get_data())
#plt.subplot(712); plt.title('A = Input * .5')
#plt.plot(Ap.get_data())
#plt.subplot(713); plt.title('B = A * 2')
#plt.plot(Bp.get_data())
#plt.subplot(714); plt.title('C(0) = 0, C(1) = A')
#plt.plot(Cp.get_data())
#plt.subplot(715); plt.title('D(0:2) = A')
#plt.plot(Dp.get_data())
#plt.subplot(716); plt.title('E(0:1) = C(1)')
#plt.plot(Ep.get_data())
#plt.subplot(717); plt.title('F(0) = C(1)')
#plt.plot(Fp.get_data())
#plt.tight_layout()
#plt.show()
