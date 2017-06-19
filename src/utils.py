'''
Created on Jun 19, 2017

@author: Varela
'''
import numpy as np 
# import collections


def umpire_at(S, U, t):
    '''
        Returns umpire locations
        S[nrounds,numpires,2] .: ndarray representing matches scheduled
        U[nrounds,numpires]   .: ndarray representing mapping from umpire to matches
        t integer or slice    .: represents a slice of time or a time = 0...nrounds
         
    '''
    
    if isinstance(t, slice):
        UI = np.array(U[t,:]-1)
        Y  = np.zeros(UI.shape, dtype=np.int32)

        ids = np.arange(S.shape[0])[t]
        
        for i, idx in enumerate(UI.tolist()):
            Y[i,:] = S[ids[i],idx,0] 
    else: 
        idx = U[t,:]-1
        Y = S[t,idx,0]
            
    return Y        

    
    
    


