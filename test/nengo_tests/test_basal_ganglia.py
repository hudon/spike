"""This is a file to create and test the basal ganglia template.
"""

import numpy as np
#import matplotlib.pyplot as plt
import math

import sys
sys.path.append(sys.argv[1])
sys.path.append(sys.argv[2])
import nef_theano as nef
import basalganglia

net = nef.Network('BG Test', seed=97)
def func(x):
    return [math.sin(x), .5,.2]
net.make_input('in', value=func)
basalganglia.make(net=net, name='BG', dimensions=3)

net.connect('in', 'BG.input')

timesteps = 1000
dt_step = 0.01
t = np.linspace(dt_step, timesteps*dt_step, timesteps)
pstc = 0.01

Ip = net.make_probe('in', dt_sample=dt_step, pstc=pstc)
BGp = net.make_probe('BG.output', dt_sample=dt_step, pstc=pstc)

print "starting simulation"
net.run(timesteps*dt_step)

ip_data = Ip.get_data()
bgp_data = BGp.get_data()


print "input 'in' probe data"
for x in ip_data:
    print x

print "bgp_data 'BG.output' probe data"
for x in bgp_data:
    print x

# plot the results
#plt.ioff(); plt.close(); 
#plt.subplot(2,1,1)
#plt.plot(t, Ip.get_data(), 'x'); plt.title('Input')
#plt.subplot(2,1,2)
#plt.plot(BGp.get_data()); plt.title('BG.output')
#plt.tight_layout()
#plt.show()
