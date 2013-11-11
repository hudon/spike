import math
def product(x):
    return x[0]*x[1]

# function example for testing
def square(x):
    return [-x[0]*x[0], -x[0], x[0]]

## basal_ganglia
# connect the striatum to the GPi and GPe (inhibitory)
def func_str(x):
    if x[0] < e: return 0
    return mm * (x[0] - e)

# connect the STN to GPi and GPe (broad and excitatory)
def func_stn(x):
    if x[0] < ep: return 0
    return mp * (x[0] - ep)

# connect the GPe to GPi and STN (inhibitory)
def func_gpe(x):
    if x[0] < ee: return 0
    return me * (x[0] - ee)

#connect GPi to output (inhibitory)
def func_gpi(x):
    if x[0]<eg: return 0
    return mg*(x[0]-eg)

def func(x):
    return [math.sin(x), .5,.2]

## eval_points
# function for testing evaluation points
def pow(x):
    return [xval**2 for xval in x]

## radius
def sin3(x):
    return math.sin(x) * 3

def mult(x):
    return [xval*2 for xval in x]

## transform
def transform_func(x):
    return [math.sin(x), -math.sin(x)]

