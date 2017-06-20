'''
Created on Jun 19, 2017

@author: Varela
'''
import numpy as np 

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
    
def umpire2segment(S, U):
    '''
        dictionary of lists of tuples
    '''            
    nrounds  = U.shape[0]  
    UI = np.array(U-1)
    u2s = {}
    for r  in xrange(nrounds):
        roundlist = S[r, UI[r,:]]
        for l, tuplelist in enumerate(roundlist):
            if u2s.has_key(l):
                u2s[l].append(tuple(tuplelist))
            else:
                u2s[l] = [tuple(tuplelist)]                    
    return u2s  
                    
    
    
    


