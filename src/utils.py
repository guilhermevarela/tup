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

def umpire2game(U):
    '''
        TUP Solution reader @ https://benchmark.gent.cs.kuleuven.be/tup/en/my_submissions/
    '''            
    nrounds, numps = U.shape
    G = np.zeros(U.shape,dtype=np.int32)
    for r in xrange(nrounds):
        for u in xrange(numps):
            gameindex = U[r,u]-1
            G[r,gameindex] = u+1 
    return G        

def umpire2homevenue(S, U):    
    nrounds, numps = U.shape
    HV = S[:,:,0]
    U2H = np.zeros(U.shape,dtype=np.int32)
    
    for r in xrange(nrounds):
        idr = U[r,:]-1
        U2H[r,:] = HV[r,idr]
    return U2H            

def violations_3_counter(S, U):
    nrounds,numpires = U.shape
    nteams  = 2*numpires  
    violations = np.zeros((numpires, nteams))
    
    for ump in xrange(numpires):
        for r in xrange(nrounds):  
            gameindex = U[r,ump]-1          
            hometeam  = S[r,gameindex,0]
            violations[ump, hometeam-1]+=1
    return  (violations.T == 0).sum(axis=0)         
             
                        
    
    
    


