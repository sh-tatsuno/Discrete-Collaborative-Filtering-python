import numpy as np
import time
from tqdm import trange
from utils import *

#discrete collaborative filtering
def DCF(init_data, alpha=0.001, beta=0.001, maxiter = 100, seed = None):
    
    if(seed): np.random.seed(seed)
        
    start_time = time.time()
    
    # init  
    ## store cell information (whether each cell is valid or null)
    Sbin = init_data["Sbin"]
    S = init_data["S"]
    r = init_data["r"]
    
    ## initialize B, D, X, Y
    m, n = S.shape
    B = init_data["B"] # r x m: user codes 
    D = init_data["D"] # r x n: item codes 
    X = init_data["X"]
    Y  = init_data["Y"]
    
    print("initialized each value")
    loss = np.sum((S - np.dot(B.T, D))**2 * Sbin)
    elapsed_time = time.time() - start_time
    print("elapsed time: " + str(elapsed_time) + "  loss: " + str(loss))

    it=0
    while(it < maxiter):
        it += 1
        print ("it: "+ str(it))
        master_flag = 0 # check for judging stop iteration

        for i in range(m):    
            while(1):
                flag = 0 # check for judging stop iteration
                bi = B[:,i].copy()
                si = S[i,:] 
                sbini = Sbin[i,:]
                for k in range(r):
                    dk = D[k,:]
                    bik = bi.copy()
                    bik[k] = 0
                    bik_hat = np.sum((si - np.dot(D.T, bik)) * dk * sbini) + alpha * X[k,i]       
                    bik_new = np.sign(K(bik_hat,  bi[k]))
                    if (bi[k] != bik_new): 
                        flag = 1
                        bi[k] = bik_new              
                if flag == 0: break
                B[:, i] = bi
                master_flag = 1

        for j in range(n):
            while(1):
                flag = 0
                dj = D[:,j].copy()
                sj = S[:,j]
                sbinj = Sbin[:,j]
                for k in range(r):
                    bk = B[k,:]
                    djk = dj.copy()
                    djk[k] = 0
                    djk_hat = np.sum((sj - np.dot(B.T, djk)) * bk * sbinj) + beta * Y[k,j]   
                    djk_new = np.sign(K(djk_hat,  dj[k]))
                    if (dj[k] != djk_new): 
                        flag = 1
                        dj[k] = djk_new 
                if flag == 0: break
                D[:, j] = dj
                master_flag = 1

        if master_flag == 0: break
            
        # Update X, Y
        X = Update_XY(B, r)
        Y  = Update_XY(D, r)
        
        # Calculate loss function
        loss = np.sum((S - np.dot(B.T, D))**2 * Sbin)
        elapsed_time = time.time() - start_time
        print("elapsed time: " + str(elapsed_time) + "  loss: " + str(loss))

    # Print total time & loss
    elapsed_time = time.time() - start_time
    loss = np.sum((S - np.dot(B.T, D))**2 * Sbin)
    print("total time: " + str(elapsed_time) + "  loss: " + str(loss))
    
    return B, D