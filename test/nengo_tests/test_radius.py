"""This is a test file to test the radius parameter of ensembles.

Need to test the radius both on identity, linear, and non-linear 
projections. It affects 3 places: the termination (scales input),
the origin (scales output), and when computing decoders (scales 
the function being computed so that it has the proper shape inside
unit length).

"""
import math

import numpy as np
#import matplotlib.pyplot as plt

import sys
sys.path.append(sys.argv[1])
import nef_theano as nef

def sin3(x):
    return math.sin(x) * 3

net = nef.Network('Encoder Test',seed=97)
net.make_input('in', value=sin3)
net.make('A', 1000, 1, radius=5)
net.make('B', 300, 1)
net.make('C', 1000, 1)
net.make('D', 300, 1)

net.connect('in', 'A')
net.connect('A', 'B')
net.connect('A', 'C')
net.connect('A', 'D')

timesteps = 0.1
dt_step = 0.01

print "starting simulation"

net.run(timesteps * dt_step)
