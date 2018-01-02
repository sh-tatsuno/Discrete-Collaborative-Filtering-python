from DCF import DCF
from utils import *
import numpy as np
import pickle
    
with open('pickle/init.pickle', 'rb') as f:
    init_data = pickle.load(f)
    
B, D = DCF(init_data, maxiter=100, seed = 43)

ret={}
ret["B"] = B
ret["D"] = D
ret["ref"] = init_data

with open('pickle/output.pickle', 'wb') as f:
    pickle.dump(ret, f)