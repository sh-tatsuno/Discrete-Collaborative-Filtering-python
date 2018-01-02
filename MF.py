import numpy as np
from tqdm import trange

# matrix factorization
# if overflow warning, you have to set alpha and beta smaller values
def MF(R, r, Rbin=None, maxsteps=200, alpha=0.001, beta=0.001, th=0.001): 
    
    m, n = R.shape
    P = np.random.rand(r, m)
    Q = np.random.rand(r, n)
    if Rbin is None:
        Rbin = (R != 0).astype('int') # binary (0 or 1) of matrix for excepting 0 value
    
    for step in trange(1, maxsteps+1):
        E = R - np.dot(P.T, Q)
        for i in range(m):
            for j in range(n):
                if Rbin[i, j] > 0:
                    P[:, i] += alpha * (2 * E[i, j] * Q[:, j] - beta * P[:, i]) 
                    Q[:, j] += alpha * (2 * E[i, j] * P[:, i] - beta * Q[:, j]) 

        #e = np.sum((R - np.dot(P.T, Q))**2 * Rbin) 
        #e += (beta / 2) * (np.sum(np.sum(P**2, axis=0) * np.sum(Rbin, axis = 1)) + np.sum(np.sum(Q**2, axis=0) * np.sum(Rbin, axis = 0)))
        #if e < th: break

    return P, Q