"""This is a test file to test basic learning
"""

import math
import time

import numpy as np
import sys
sys.path.append(sys.argv[1])
import nef_theano as nef
#import matplotlib.pyplot as plt

neurons = 30  # number of neurons in all ensembles

net = nef.Network('Learning Test',seed=72)
net.make_input('in', value=0.8)
timer = time.time()
net.make('A', neurons=neurons, dimensions=1)
net.make('B', neurons=neurons, dimensions=1)
net.make('error1', neurons=neurons, dimensions=1)
print "Made populations:", time.time() - timer

net.learn(pre='A', post='B', error='error1')

net.connect('in', 'A')
net.connect('A', 'error1')
net.connect('B', 'error1', weight=-1)

t_final = 10
dt_step = 0.01
pstc = 0.03

Ip = net.make_probe('in', dt_sample=dt_step, pstc=pstc)
Ap = net.make_probe('A', dt_sample=dt_step, pstc=pstc)
Bp = net.make_probe('B', dt_sample=dt_step, pstc=pstc)
E1p = net.make_probe('error1', dt_sample=dt_step, pstc=pstc)

print "starting simulation"
net.run(t_final)

#plt.ioff(); plt.close()

ip_data = Ip.get_data()
ap_data = Ap.get_data()
bp_data = Bp.get_data()
e1p_data = E1p.get_data()

print "input 'in' probe data"
for x in ip_data:
    print x

print "input 'ap' probe data"
for x in ap_data:
    print x

print "input 'bp' probe data"
for x in bp_data:
    print x

print "input 'e1p' probe data"
for x in e1p_data:
    print x

t = np.linspace(0, t_final, len(Ap.get_data()))

#plt.plot(t, Ap.get_data())
#plt.plot(t, Bp.get_data())
#plt.plot(t, E1p.get_data())
#plt.legend(['A', 'B', 'error'])
#plt.title('Normal learning')
#plt.tight_layout()
#plt.show()
