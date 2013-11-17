"""Test of Network.make_subnetwork(), which should allow for easy nesting of
collections of ensembles.
"""

import numpy as np
#import matplotlib.pyplot as plt

import sys
sys.path.append(sys.argv[1])
import nef_theano as nef

hosts_file = sys.argv[2] if len(sys.argv) > 2 else None
if hosts_file:
  net = nef.Network('Main', seed=93, hosts_file=hosts_file)
else:
  net = nef.Network('Main', seed=93)

netA = net.make_subnetwork('A')
netB = net.make_subnetwork('B')

net.make('X',50,1)
netA.make('Y',50,1)
netB.make('Z',50,1)
netB.make('W',50,1)

netB.connect('Z','W')     # connection within a subnetwork
net.connect('X','A.Y')    # connection into a subnetwork
net.connect('A.Y','X')    # connection out of a subnetwork
net.connect('A.Y','B.Z')  # connection across subnetworks

netC = netA.make_subnetwork('C')
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
Yp = net.make_probe('A.Y', dt_sample=dt_step, pstc=pstc)
Zp = net.make_probe('B.Z', dt_sample=dt_step, pstc=pstc)
Wp = net.make_probe('B.W', dt_sample=dt_step, pstc=pstc)
Ip = net.make_probe('A.C.I', dt_sample=dt_step, pstc=pstc)
Jp = net.make_probe('A.C.J', dt_sample=dt_step, pstc=pstc)


net.run(times * dt_step) # run for 1 second

xp_data = Xp.get_data()
yp_data = Yp.get_data()
zp_data = Zp.get_data()
wp_data = Wp.get_data()
ip_data = Ip.get_data()
jp_data = Jp.get_data()

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
