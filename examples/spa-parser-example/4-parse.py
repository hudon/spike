import spa

import sys
sys.path.append(sys.argv[1])
import nef_theano as nef

dimensions = 8   # should be 64

class Rules:
    def verb(vision='WRITE'):
        effect(verb=vision)
    def noun(vision='ONE+TWO+THREE'):
        effect(noun=vision)
    def write(vision='0.5*NONE-0.5*WRITE-0.5*ONE-0.5*TWO-0.5*THREE', phrase='0.5*WRITE*VERB'):
        effect(motor=phrase*'~NOUN')

class ParseWrite(spa.SPA):
    verbose = True

    vision = spa.Buffer(dimensions=dimensions, feedback=0)
    phrase = spa.Buffer(dimensions=dimensions, feedback=0)
    motor = spa.Buffer(dimensions=dimensions, feedback=0)

    noun = spa.Buffer(dimensions=dimensions)
    verb = spa.Buffer(dimensions=dimensions)



    BG = spa.BasalGanglia(Rules)
    thal = spa.Thalamus(BG)

    input = spa.Input(0.5, vision='WRITE')
    input.next(0.5, vision='ONE')
    input.next(0.5, vision='NONE')


net = nef.Network('ParseWrite', seed=1, command_arguments=sys.argv[2:],
        usr_module='../examples/spa-example/functions.py')

pw = ParseWrite(net)

net.connect('noun.buffer', 'phrase.buffer',
    transform=pw.sinks['phrase'].parse('NOUN').get_transform_matrix())
net.connect('verb.buffer', 'phrase.buffer',
    transform=pw.sinks['phrase'].parse('VERB').get_transform_matrix())


pThal = net.make_probe('thal.rule', dt_sample=0.001)
pMotor = net.make_probe('motor.buffer', dt_sample=0.001)

net.run(1.5)

import numpy as np
import pylab
pylab.figure()
pylab.plot(pThal.get_data())
pylab.figure()
v = pw.sinks['motor']
pylab.plot(np.dot(pMotor.get_data(), v.vectors.T))
pylab.show()
