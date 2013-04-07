import sys
#  This will allow us to dynamically set the location of the nef package we want to test
sys.path.append(sys.argv[1])
import nef

net=nef.Network('Function Test',seed=3)
net.make_input('input',0.5)
net.make('A',100,1)
net.connect('input','A')
net.make('B',100,3)

def square(x):
    return x[0]*x[0],-x[0],x[0]

net.connect('A','B',func=square,pstc=0.1)

for i in range(10):
	net.run(0.001)
	if hasattr(net, 'node'):
		print i,net.node['B'].accumulator[0.1].value.get_value()
	else:
		print i,net.nodes['B'].accumulator[0.1].value.get_value()

if hasattr(net, 'clean_up'):
	net.clean_up()
