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
    return  umpire_sawteam(S, U, t, iteam=0)

def umpire_sawteam(S, U, t, iteam=0):
    '''
        Slices S (schedule) in order to team in a match 
        S[nrounds,numpires,2] .: ndarray representing matches scheduled
        U[nrounds,numpires]   .: ndarray representing mapping from umpire to matches
        t integer or slice    .: represents a slice of time or a time = 0...nrounds
        iteam                 .: item=0 hometeam, item=1, advteam 
         
    '''
    
    if isinstance(t, slice):
        UI = np.array(U[t,:]-1)
        Y  = np.zeros(UI.shape, dtype=np.int32)

        ids = np.arange(S.shape[0])[t]
        
        for i, idx in enumerate(UI.tolist()):
            Y[i,:] = S[ids[i],idx,iteam] 
    else: 
        idx = U[t,:]-1
        Y = S[t,idx,iteam]
            
    return Y        
    
    
    


