import sys
#  This will allow us to dynamically set the location of the nef package we want to test
sys.path.append(sys.argv[1])
import nef
import numpy
net=nef.Network('Cleanup',seed=3)

D=128
M=50000
N1=100
N2=50
index=0

numpy.random.seed(2)
def make_vector():
    v=numpy.random.normal(size=(D,))
    
    norm=numpy.linalg.norm(v)
    v=v/norm
    return v


print 'making words...'
words=[make_vector() for i in range(M)]
words=numpy.array(words)
print '...done'

net.make_array('A',N1,D)
net.make_array('B',N1,D)

net.make_array('C',N2,M)#,intercept=(0.6,0.9))
print 'made'

net.connect('A','C',words.T,pstc=0.1)
net.connect('C','B',words,pstc=0.1)


net.make_input('input',words[index])
net.connect('input','A',pstc=0.1)

for i in range(1):
	net.run(0.001)
	if hasattr(net, 'node'):
		print i,net.node['A'].origin['X'].value.get_value()
	else:
		print i,net.nodes['A'].origin['X'].value.get_value()

if hasattr(net, 'clean_up'):
	net.clean_up()
