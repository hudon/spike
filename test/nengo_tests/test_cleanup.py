import nose
import numpy as np

import sys
sys.path.append(sys.argv[1])
import nef_theano as nef


def test_cleanup():

    net = nef.Network('Cleanup',seed=3)

    D = 5
    M = 5
    N1 = 1
    N2 = 5
    index = 0

    def make_vector():
        v = np.random.normal(size=(D,))
        
        norm = np.linalg.norm(v)
        v = v / norm
        return v

    print 'making words...'
    words = [make_vector() for i in range(M)]
    words = np.array(words)
    print '...done'

    net.make_array('A', N1, D)
    net.make_array('B', N1, D)

    net.make_array('C', N2, M)#,intercept=(0.6,0.9))
    print 'made'

    net.connect('A', 'C', words.T, pstc=0.1)
    net.connect('C', 'B', words, pstc=0.1)

    net.make_input('input', words[index])
    net.connect('input', 'A', pstc=0.1)

    Ap = net.make_probe('A', dt_sample=0.001, pstc=0.1)
    Bp = net.make_probe('B', dt_sample=0.001, pstc=0.1)
    Cp = net.make_probe('C', dt_sample=0.001, pstc=0.1)

    for i in range(10):
        net.run(0.001)

    ap_data = Ap.get_data()
    bp_data = Bp.get_data()
    cp_data = Cp.get_data()

    print "ensemble 'A' probe data"
    for x in ap_data:
        print x
    print "ensemble 'B' probe data"
    for x in bp_data:
        print x
    print "ensemble 'C' probe data"
    for x in cp_data:
        print x
	
test_cleanup()
