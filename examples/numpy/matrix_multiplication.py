# Perform matrix multiplication on arbitrary matrices

from ccm.lib import nef
import numpy


# Adjust these values to change the matrix dimensions
#  Matrix A is D1xD2
#  Matrix B is D2xD3
#  result is D1xD3

D1=5
D2=5
D3=5
pstc=0.01
dt=0.001


A=[nef.ScalarNode() for i in range(D1*D2)]
B=[nef.ScalarNode() for i in range(D2*D3)]
for n in A+B:
    n.configure(neurons=50)
    n.configure_spikes(pstc=pstc,dt=dt)



# the C matrix holds the intermediate product calculations
#  need to compute D1*D2*D3 products to multiply 2 matrices together

C=[nef.VectorNode(2) for i in range(D1*D2*D3)]
for n in C:
    n.configure(neurons=200)
    n.configure_spikes(pstc=pstc,dt=dt)
    

# determine the transformation matrices to get the correct pairwise
#  products computed.  This looks a bit like black magic but if
#  you manually try multiplying two matrices together, you can see
#  the underlying pattern.  Basically, we need to build up D1*D2*D3
#  pairs of numbers in C to compute the product of.  If i,j,k are the
#  indexes into the D1*D2*D3 products, we want to compute the product
#  of element (i,j) in A with the element (j,k) in B.  The index in
#  A of (i,j) is j+i*D2 and the index in B of (j,k) is k+j*D3.
for i in range(D1):
    for j in range(D2):
        for k in range(D3):            
            A[j+i*D2].connect(C[(j+k*D2+i*D2*D3)],weight=[1,0])
            B[k+j*D3].connect(C[(j+k*D2+i*D2*D3)],weight=[0,1])
                        
            

D=[nef.ScalarNode() for i in range(D1*D2)]
for n in D:
    n.configure(neurons=50)
    n.configure_spikes(pstc=pstc,dt=dt)


def product(x):
    return x[0]*x[1]
# the mapping for this transformation is much easier, since we want to
# combine D2 pairs of elements (we sum D2 products together)    

for i in range(D1*D2*D3):
    C[i].connect(D[i/D2],func=product)

print 'neurons:',50*(D1*D2+D2*D3+D1*D3)+200*(D1*D2*D3)
A[0].tick(dt=dt)
import time
start=time.time()
for i in range(5000):
    A[0].tick(dt=dt)
    print "time per tick:",(time.time()-start)/(i+1)

