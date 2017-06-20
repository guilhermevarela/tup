'''
Created on Jun 8, 2017

@author: Varela
'''
import numpy as np 
from utils import umpire_at, umpire_sawteam

def travel_builder(D, S, U, t):
    # Builds a matrix for maximum flow algorithm     
    nmatches    = S.shape[1] 
    origin      = U[t-1,:]   # From last UMPIRE LOCATIONS
    destination = S[t,:, 0]   # From next GAME LOCATIONS
    
    #index with all combinations
    R = np.tile((origin-1).reshape(nmatches,1),(1,nmatches))
    C = np.tile((destination-1).reshape(1,nmatches),(nmatches,1)) 
         
    Tt = D[R, C].T   
    return Tt

             
def schedule_builder(opponents):
    # Converts the 2D opponents matrix from files to 3D
    nrounds, nteams = opponents.shape
    newshape = (nrounds,int(nteams/2))
    
    #home index are those index that have a positive value
    I = np.tile(np.arange(nteams).reshape(1,nteams),(nrounds,1))
    homerows, homecols  = np.where(opponents > 0)
    H = I[homerows,homecols]+1         # from index to teams
    H = H.reshape(newshape)            #official schedule 
      
    #visiting index are those index that have a negative value
    maskv  = opponents > 0    # from index to teams
    V      = opponents[maskv]     # linearized
    V      = V.reshape(newshape)
    
    S     = np.zeros((nrounds,int(nteams/2),2), dtype=np.int32)
    S[:,:,0]     = H 
    S[:,:,1]     = V 
    return S

def umpires_builder(nrounds,nteams):
    return np.zeros((nrounds, int(nteams/2)), dtype=np.int32)

def constraint_4_builder(D, S, U, t, d1):  
    numpires = S.shape[1]
    nmatches = numpires    
    nvenues  = 2*numpires 
    Cmask     = np.zeros((numpires,nvenues),dtype=np.int32)
    idumpires = np.arange(numpires).reshape(1,numpires)
    
    
    if t > 0:                
          
        # MAPS LOCATIONS TO UMPIRES
        y           = max(t-(numpires-d1),0)        
        s = slice(y,t)
        L = umpire_at(S, U, s)
        #Constraints 
        LI = (L - 1)                                    #COLS INDEX         
        for idg in LI.tolist():
            Cmask[idumpires,idg] = 1        
        
        idt  = (S[t,:,0].reshape((numpires,))-1)

        C4t = Cmask[:,idt]         
                                   
    return C4t

def constraint_5_builder(D, S, U, t, d2):  
    numpires = S.shape[1]
    nmatches = numpires    
    nvenues  = 2*numpires 
    Cmask     = np.zeros((numpires,nvenues),dtype=np.int32)
    idumpires = np.arange(numpires).reshape(1,numpires)
    
    
    if t > 0:                          
        # MAPS LOCATIONS TO UMPIRES
        y = max(t-(int(numpires/2)-d2),0)        
        s = slice(y,t)
        for team in xrange(2):
            L = umpire_sawteam(S, U, s, team=team)
            #Constraints 
            LI = (L - 1)              
            for idg in LI.tolist():
                Cmask[idumpires,idg] = 1
                            
        idt  = (S[t,:,0].reshape((numpires,))-1)

        C5t = Cmask[:,idt]         
                                           
    return C5t                     
         
    