# Perform matrix multiplication on arbitrary matrices
import nef
import numpy

net = nef.Network('Matrix Multiplication', seed=1)  # Create the network object

# Adjust these values to change the matrix dimensions
#  Matrix A is D1xD2
#  Matrix B is D2xD3
#  result is D1xD3

D1 = 5
D2 = 5
D3 = 5

# make 2 matrices to store the input (ensembles?)
net.make_array(name='A', neurons=50, count=D1 * D2)
net.make_array(name='B', neurons=50, count=D2 * D3)

# connect inputs to them so we can set their value
net.make_input(name='input A', value=[0] * D1 * D2)
net.make_input(name='input B', value=[0] * D2 * D3)
net.connect(pre='input A', post='A')
net.connect(pre='input B', post='B')

net.make_array('C', 200, D1 * D2 * D3, dimensions=2,
               encoders=[[1, 1], [1, -1], [-1, 1], [-1, -1]])

transformA = [[0] * (D1 * D2) for i in range(D1 * D2 * D3 * 2)]
transformB = [[0] * (D2 * D3) for i in range(D1 * D2 * D3 * 2)]
for i in range(D1):
    for j in range(D2):
        for k in range(D3):
            transformA[(j + k * D2 + i * D2 * D3) * 2][j + i * D2] = 1
            transformB[(j + k * D2 + i * D2 * D3) * 2 + 1][k + j * D3] = 1

transformE = [[0] * (D3 * D1) for i in range(D1 * D3 * D1 * 2)]
transformD = [[0] * (D1 * D3) for i in range(D1 * D3 * D1 * 2)]
for i in range(D1):
    for j in range(D3):
        for k in range(D1):
            transformE[(j + k * D3 + i * D3 * D1) * 2][j + i * D3] = 1
            transformD[(j + k * D3 + i * D3 * D1) * 2 + 1][k + j * D1] = 1

net.connect('A', 'C', transform=numpy.array(transformA).T)
net.connect('B', 'C', transform=numpy.array(transformB).T)

# now compute the products and do the appropriate summing
net.make_array('D', 50, D1 * D3, type='lif-rate')

def product(x):
    return x[0] * x[1]
# the mapping for this transformation is much easier, since we want to
# combine D2 pairs of elements (we sum D2 products together)

transform = [[0] * (D1 * D3) for i in range(D1 * D2 * D3)]
for i in range(D1 * D2 * D3):
    transform[i][i / D2] = 1

transformG = [[0] * (D1 * D1) for i in range(D1 * D3 * D1)]
for i in range(D1 * D3 * D1):
    transformG[i][i / D3] = 1

net.connect('C', 'D', transform=transform, func=product)

net.make_input(name='input E', value=[0] * D3 * D1)
net.make_array(name='E', neurons=50, count=D3 * D1)

net.make_array(name='F', neurons=200, count=D1 * D3 * D1, dimensions=2,
               encoders=[[1, 1], [1, -1], [-1, 1], [-1, -1]])

net.make_array(name='G', neurons=50, count=D1 * D1, type='lif-rate')

net.connect(pre='input E', post='E')
net.connect('E', 'F', transform=numpy.array(transformE).T)
net.connect('D', 'F', transform=numpy.array(transformD).T)
net.connect('F', 'G', transform=transformG, func=product)

print 'neurons:', 50 * (D1 * D2 + D2 * D3 + D1 * D3) + 200 * (D1 * D2 \
        * D3) + 50 * 3 * (D1 * D3)
net.run(0.001)
import time
start = time.time()
for i in range(5000):
    net.run(0.001)
    print "time per tick:", (time.time() - start) / (i + 1)
net.clean_up()
