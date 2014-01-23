
eg = -0.2
mg = 1
def func_gpi(x):
    if x[0] < eg: return 0
    return mg * (x[0] - eg)

e = 0.2
mm = 1

# to connect the striatum to the GPi and GPe (inhibitory)
def func_str(x):
    if x[0] < e: return 0
    return mm * (x[0] - e)

ep=-0.25
mp=1
# to connect the STN to GPi and GPe (broad and excitatory)
def func_stn(x):
    if x[0] < ep: return 0
    return mp * (x[0] - ep)

# connect the GPe to GPi and STN (inhibitory)
ee=-0.2
me=1
def func_gpe(x):
    if x[0] < ee: return 0
    return me * (x[0] - ee)

#def result(x,v=self.spa.sinks[post[0]].parse(post[1]).v):
def result(x,v=[]):
    for xx in x:
        if xx<0.4: return [0]*len(v)  #TODO: This is pretty arbitrary....
    return v
