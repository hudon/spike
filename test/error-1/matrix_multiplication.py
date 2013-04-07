import sys
#  This will allow us to dynamically set the location of the nef package we want to test
sys.path.append(sys.argv[1])
# Perform matrix multiplication on arbitrary matrices

import nef
import numpy

net = nef.Network('Matrix Multiplication', seed = 1) #Create the network object

# Adjust these values to change the matrix dimensions
#  Matrix A is D1xD2
#  result is D1xD3

D1 = 1
D2 = 1
D3 = 1

# make 2 matrices to store the input
net.make_array('A', 50, D1 * D2)

# connect inputs to them so we can set their value
net.make_input('input A', [0] * D1 * D2)
net.connect('input A', 'A')


# the C matrix holds the intermediate product calculations
#  need to compute D1*D2*D3 products to multiply 2 matrices together
net.make_array('C', 200, D1 * D2 * D3, dimensions = 2,
    encoders = [[1, 1], [1, -1], [-1, 1], [-1, -1]])

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
for i in range(D1):
    for j in range(D2):
        for k in range(D3):
            transformA[(j + k * D2 + i * D2 * D3) * 2][j + i * D2] = 1

net.connect('A', 'C', transform = numpy.array(transformA).T)


print 'neurons:', 50 * (D1 * D2 + D2 * D3 + D1 * D3) + 200 * (D1 * D2 * D3)
import time
start = time.time()
for i in range(10):
	print "Iteration ",i," ",time.time()
	net.run(0.001)

if hasattr(net, 'clean_up'):
	net.clean_up()
