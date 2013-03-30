# Perform matrix multiplication on arbitrary matrices

import nef
import numpy

## 0MQ stuff (for when we port from threading to cluster distributing
#import zmq
#context = zmq.Context()
# TODO get make_array to distribute 'a', 'b', 'c'
# TODO get connect() to setup the 0MQ connections
# TODO get update() to do the communication

net = nef.Network('Matrix Multiplication', seed=1)  # Create the network object

# Adjust these values to change the matrix dimensions
#  Matrix A is D1xD2
#  Matrix B is D2xD3
#  result is D1xD3

D1 = 1
D2 = 5
D3 = 5

# make 2 matrices to store the input (ensembles?)
# NOTE: the 50 is hardcoded here, shouldn't it depend on D1 * D2?
# NOTE: answer: no, neurons is the number of neurons per ensemble, count is
# the number of arrays in the ensemble array (which is called "Ensemble" here)
net.make_array(name='A', neurons=50, count=D1 * D2)
B = net.make_array(name='B', neurons=200, count=D2 * D3, isprocess=False)
encoder_length = len(B.encoders)
decoder_length = len(B.decoders)
B1 = net.make_array(name='B1', neurons=100, count=D2 * D3,
               encoders=B.encoders[:encoder_length / 2], override_encoders=True,
               decoders=B.decoders[:decoder_length / 2])
B2 = net.make_array(name='B2', neurons=100, count=D2 * D3,
               encoders=B.encoders[encoder_length / 2:], override_encoders=True,
               decoders=B.decoders[decoder_length / 2:])


# connect inputs to them so we can set their value
net.make_input(name='input A', value=[0] * D1 * D2)
net.make_input(name='input B', value=[0] * D2 * D3)
net.connect(pre='input A', post='A')
net.connect(pre='input B', post='B1')
net.connect(pre='input B', post='B2')

print "B encoders", B.encoders
print "B decoders", B.decoders
print "B1 encoders", B1.encoders
print "B1 decoders", B1.decoders
print "B2 encoders", B2.encoders
print "B2 decoders", B2.decoders

# the C matrix holds the intermediate product calculations
#  need to compute D1*D2*D3 products to multiply 2 matrices together
net.make_array('C', 200, D1 * D2 * D3, dimensions=2,
               encoders=[[1, 1], [1, -1], [-1, 1], [-1, -1]])

# determine the transformation matrices to get the correct pairwise
#  products computed.  This looks a bit like black magic but if
#  you manually try multiplying two matrices together, you can see
#  the underlying pattern.  Basically, we need to build up D1*D2*D3
#  pairs of numbers in C to compute the product of.  If i,j,k are the
#  indexes into the D1*D2*D3 products, we want to compute the product
#  of element (i,j) in A with the element (j,k) in B.  The index in
#  A of (i,j) is j+i*D2 and the index in B of (j,k) is k+j*D3.
#  The index in C is j+k*D2+i*D2*D3, multiplied by 2 since there are
#  two values per ensemble.  We add 1 to the B index so it goes into
#  the second value in the ensemble.
transformA = [[0] * (D1 * D2) for i in range(D1 * D2 * D3 * 2)]
transformB = [[0] * (D2 * D3) for i in range(D1 * D2 * D3 * 2)]
for i in range(D1):
    for j in range(D2):
        for k in range(D3):
            transformA[(j + k * D2 + i * D2 * D3) * 2][j + i * D2] = 1
            transformB[(j + k * D2 + i * D2 * D3) * 2 + 1][k + j * D3] = 1

net.connect('A', 'C', transform=numpy.array(transformA).T)
net.connect('B1', 'C', transform=numpy.array(transformB).T)
net.connect('B2', 'C', transform=numpy.array(transformB).T)


# now compute the products and do the appropriate summing
net.make_array('D', 50, D1 * D3, type='lif-rate')


def product(x):
    return x[0] * x[1]
# the mapping for this transformation is much easier, since we want to
# combine D2 pairs of elements (we sum D2 products together)

transform = [[0] * (D1 * D3) for i in range(D1 * D2 * D3)]
for i in range(D1 * D2 * D3):
    transform[i][i / D2] = 1

net.connect('C', 'D', transform=transform, func=product)

print 'neurons:', 50 * (D1 * D2 + D2 * D3 + D1 * D3) + 200 * (D1 * D2 * D3)
net.run(0.001)
import time
start = time.time()
for i in range(1000):
    net.run(0.001)
    print "time per tick:", (time.time() - start) / (i + 1)
#net.clean_up()
import sys
print >> sys.stderr, "done"
