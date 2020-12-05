import numpy as np 

def odctdict(N, L):
    """
    @brief      Creates an overcomplete DCT dictionary
    @param      N   patch size
    @param      L   dictionary size
    @return     overcomplete DCT dictionary of shape (NxL)
    """
    D = np.zeros((N, L))
    D[:,0] = 1/np.sqrt(N)
    for k in range(1, L):
        v = np.cos(np.arange(N) * np.pi * (k)/L)
        v = v - np.mean(v)
        D[:,k] = v / np.linalg.norm(v)
    return D
