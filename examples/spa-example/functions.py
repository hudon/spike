
eg = -0.2
mg = 1
def get_output_function(self):    
    def func_gpi(x):
        if x[0] < eg: return 0
        return mg * (x[0] - eg)
    return func_gpi

e = 0.2
mm = 1

def func_str(x):
    if x[0] < e: return 0
    return mm * (x[0] - e)
