import spa
import sys
sys.path.append(sys.argv[1])
import nef_theano as nef

class Rules:
    def A(state='A'):
        effect(state='B')
    def B(state='B'):
        effect(state='C')
    def C(state='C'):
        effect(state='D')
    def D(state='D'):
        effect(state='E')
    def E(state='E'):
        effect(state='A')
    
class Sequence(spa.SPA):
    dimensions = 8
    verbose = True
    
    state = spa.Buffer()
    BG = spa.BasalGanglia(Rules)
    thal = spa.Thalamus(BG)
    
    input = spa.Input(0.1,state='D')

net = nef.Network('Sequence', seed=1, command_arguments=sys.argv[2:])
seq = Sequence(net)

pThal = net.make_probe('thal.rule', dt_sample=0.001)

net.run(1)

import pylab
pylab.plot(pThal.get_data())
pylab.show()
