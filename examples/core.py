import numpy
import math
import sys


class FPGA:
    def __init__(self,data):
        self.N=len(data)
        self.D=(len(data[0])-2)/2
        print self.N,self.D

        self.dt=0.001
        self.pstc=0.05
        
        data=numpy.array(data)
        self.encoders=data[:,2:(2+self.D)]
        self.decoders=data[:,(2+self.D):]/self.dt
        
        self.state=numpy.zeros(self.D)
        self.voltage=numpy.zeros(self.N)
        self.refractory_time=numpy.zeros(self.N)
        self.J_bias=data[:,1]
        self.Jm_prev=None
        self.t_rc=0.01
        self.t_ref=0.001
        
        
    def tick(self):
        Jm=numpy.dot(self.encoders,self.state)+self.J_bias
        dt=self.dt
        if self.Jm_prev is None: self.Jm_prev=Jm
        v=self.voltage

        # Euler's method
        dV=dt/self.t_rc*(self.Jm_prev-v)

        self.Jm_prev=Jm
        v+=dV
        v=numpy.maximum(v,0)
        
        post_ref=2-self.refractory_time/dt
        
        v*=numpy.clip(post_ref,0,1)
        
        V_threshold=1
        spiked=numpy.where(v>V_threshold,1,0)

        overshoot=(v-V_threshold)/dV
        spiketime=dt*(1.0-overshoot)
        self.refractory_time=numpy.where(spiked,spiketime+self.t_ref,self.refractory_time-dt)
        
        self.voltage=v*(1-spiked)
        
        new_state=numpy.dot(spiked,self.decoders)
        
        # apply the filter
        decay=math.exp(-dt/self.pstc)
        self.state=self.state*decay+(1-decay)*new_state
              
        
        
def read_data(filename):
    for line in open(filename).readlines():
        yield [float(x) for x in line.strip().split(',')]            

f=FPGA(list(read_data('communicate.csv')))
    
    
import time
n_iters = 1000
print 'Running %i iterations' % n_iters
t0 = time.time()
for j in range(1000):
    f.state[:f.D/3]=0.5
    f.tick()
print 'Time per iteration', ((time.time() - t0) / n_iters)
print 'Final state vector', f.state
    
    
