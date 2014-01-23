import numpy as np
import matplotlib.pyplot as plt
import math

import nengo_theano as nef


# DxD discrete fourier transform matrix            
def discrete_fourier_transform(D):
    m=[]
    for i in range(D):
        row=[]
        for j in range(D):            
            row.append(complex_exp((-2*math.pi*1.0j/D)*(i*j)))
        m.append(row)
    return m

# DxD discrete inverse fourier transform matrix            
def discrete_fourier_transform_inverse(D):
    m=[]
    for i in range(D):
        row=[]
        for j in range(D):            
            row.append(complex_exp((2*math.pi*1.0j/D)*(i*j))/D)
        m.append(row)
    return m

# formula for e^z for complex z
def complex_exp(z):
    a=z.real
    b=z.imag
    return math.exp(a)*(math.cos(b)+1.0j*math.sin(b))

def product(x):
    return x[0]*x[1]


def output_transform(dimensions):
    ifft=np.array(discrete_fourier_transform_inverse(dimensions))

    def makeifftrow(D,i):
        if i==0 or i*2==D: return ifft[i]
        if i<=D/2: return ifft[i]+ifft[-i].real-ifft[-i].imag*1j
        return np.zeros(dimensions)
    ifftm=np.array([makeifftrow(dimensions,i) for i in range(dimensions/2+1)])
    
    ifftm2=[]
    for i in range(dimensions/2+1):
        ifftm2.append(ifftm[i].real)
        ifftm2.append(-ifftm[i].real)
        ifftm2.append(-ifftm[i].imag)
        ifftm2.append(-ifftm[i].imag)
    ifftm2=np.array(ifftm2)

    return ifftm2.T

def input_transform(dimensions,first,invert=False):
    fft=np.array(discrete_fourier_transform(dimensions))

    M=[]
    for i in range((dimensions/2+1)*4):
        if invert: row=fft[-(i/4)]
        else: row=fft[i/4]
        if first:
            if i%2==0:
                row2=np.array([row.real,np.zeros(dimensions)])
            else:
                row2=np.array([row.imag,np.zeros(dimensions)])
        else:
            if i%4==0 or i%4==3:
                row2=np.array([np.zeros(dimensions),row.real])
            else:    
                row2=np.array([np.zeros(dimensions),row.imag])
        M.extend(row2)
    return M

def circconv(a, b):
    return np.fft.ifft(np.fft.fft(a)*np.fft.fft(b)).real

D = 16
subD = 8
N = 50
N_C = 200

net=nef.Network('Main', fixed_seed=1)

A = np.random.randn(D)
A /= np.linalg.norm(A)

B = np.random.randn(D)
B /= np.linalg.norm(B)



net.make_input('inA', A) 
net.make_input('inB', B) 


net.make_array('A', N*subD, D/subD, dimensions=subD)
net.make_array('B', N*subD, D/subD, dimensions=subD)
net.make_array('D', N*subD, D/subD, dimensions=subD)

net.make_array('C', N_C, (D/2+1)*4, dimensions=2, 
                   encoders=[[1,1],[1,-1],[-1,1],[-1,-1]], radius=3)
                   
AT = input_transform(D, True, False)
BT = input_transform(D, False, False)
    
net.connect('A', 'C', transform=AT, pstc=0.01)
net.connect('B', 'C', transform=BT, pstc=0.01)
                   
ifftm2=output_transform(D)
    
net.connect('C', 'D', func=product, transform=ifftm2, pstc=0.01)

net.connect('inA', 'A')
net.connect('inB', 'B')



net.run(1.0)
    


