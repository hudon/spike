"""This test file is for checking the run time of the theano code."""

import math
import time

import sys
sys.path.append(sys.argv[1])
import nef_theano as nef

import functions

net = nef.Network('Runtime Test', seed=123, command_arguments=sys.argv[2:], usr_module='test/nengo_tests/functions.py')

net.make_input('in', value=math.sin)
net.make('A', neurons=1000, dimensions=1)
net.make('B', neurons=1000, dimensions=1)
net.make('C', neurons=1000, dimensions=1)
net.make('D', neurons=1000, dimensions=1)

net.connect('in', 'A')
net.connect('A', 'B')
net.connect('A', 'C', func=functions.pow)
net.connect('A', 'D', func=functions.mult)
net.connect('D', 'B', func=functions.pow) # throw in some recurrency whynot

dt_step=0.01
pstc=0.01

Ip = net.make_probe('in', dt_sample=dt_step, pstc=pstc)
Ap = net.make_probe('A', dt_sample=dt_step, pstc=pstc)
Bp = net.make_probe('B', dt_sample=dt_step, pstc=pstc)
Cp = net.make_probe('C', dt_sample=dt_step, pstc=pstc)
Dp = net.make_probe('D', dt_sample=dt_step, pstc=pstc)

print "starting simulation"
net.run(0.5)

ip_data = Ip.get_data()
ap_data = Ap.get_data()
bp_data = Bp.get_data()
cp_data = Cp.get_data()
dp_data = Dp.get_data()

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
