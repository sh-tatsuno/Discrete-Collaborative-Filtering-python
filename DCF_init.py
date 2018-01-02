from MF import MF
from Process_MovieLens import *
from utils import *
import numpy as np
import pickle
    
ret = {}
rating_matrix, items = load_data()

r = 20
init_alpha = 10e-6
init_beta = 10e-6

# init  
## store cell information (whether each cell is valid or null)
S_ = rating_matrix
Sbin = (S_ > 0).astype('int') # binary (0 or 1) of matrix for excepting 0 value

## scale S
S = (S_ - 1) / np.max(S_) # S ~ [0, 1] . Assume that minimun value is 1.
S = 2 * r * S - r # S ~ [-r, r]

## initialize B, D, X, Y
m, n = S.shape
B, D = MF(S, r, Sbin, maxsteps=200, alpha=init_alpha, beta=init_beta)

ret['rating_matrix']=rating_matrix
ret['items']=items
ret['S']=S
ret['r']=r
ret['Sbin']=Sbin
ret['B_MF']=B
ret['D_MF']=D

B = np.sign(B) # r x m: user codes 
D = np.sign(D) # r x n: item codes 
X = Update_XY(B, r)
Y  = Update_XY(D, r)

ret['B']=B
ret['D']=D
ret['X']=X
ret['Y']=Y

with open('pickle/init.pickle', 'wb') as f:
    pickle.dump(ret, f)