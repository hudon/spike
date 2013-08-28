"""Test of Network.make_subnetwork(), which should allow for easy nesting of
collections of ensembles.
"""

import numpy as np
#import matplotlib.pyplot as plt

import sys
sys.path.append(sys.argv[1])
import nef_theano as nef

net=nef.Network('Main', seed=93)

netA=net.make_subnetwork('A')
netB=net.make_subnetwork('B')

net.make('X',50,1)
netA.make('Y',50,1)
netB.make('Z',50,1)
netB.make('W',50,1)

netB.connect('Z','W')     # connection within a subnetwork
net.connect('X','A.Y')    # connection into a subnetwork
net.connect('A.Y','X')    # connection out of a subnetwork
net.connect('A.Y','B.Z')  # connection across subnetworks

netC=netA.make_subnetwork('C')
netC.make('I',50,1)
netC.make('J',50,1)
netC.connect('I','J')       # connection within a subsubnetwork
net.connect('X','A.C.I')    # connection into a subsubnetwork
net.connect('A.C.J','X')    # connection out of a subsubnetwork
net.connect('A.C.J','B.Z')  # connection across subsubnetworks
netA.connect('Y','C.J')     # connection across subnetworks

dt_step = 0.01
pstc = 0.01
times = 100
Xp = net.make_probe('X', dt_sample=dt_step, pstc=pstc)
xp_data = Xp.get_data()

Yp = net.make_probe('A.Y', dt_sample=dt_step, pstc=pstc)
yp_data = Yp.get_data()

Zp = net.make_probe('B.Z', dt_sample=dt_step, pstc=pstc)
zp_data = Zp.get_data()

Wp = net.make_probe('B.W', dt_sample=dt_step, pstc=pstc)
wp_data = Wp.get_data()

Ip = net.make_probe('A.C.I', dt_sample=dt_step, pstc=pstc)
ip_data = Ip.get_data()

Jp = net.make_probe('A.C.J', dt_sample=dt_step, pstc=pstc)
jp_data = Jp.get_data()

net.run(times * dt_step) # run for 1 second

print "ensemble 'X' probe data"
for x in xp_data:
    print x

print "ensemble 'Y' probe data"
for x in yp_data:
    print x

print "ensemble 'Z' probe data"
for x in zp_data:
    print x

print "ensemble 'W' probe data"
for x in wp_data:
    print x

print "ensemble 'I' probe data"
for x in ip_data:
    print x

print "ensemble 'J' probe data"
for x in jp_data:
    print x







