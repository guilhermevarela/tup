'''
Created on Jun 8, 2017

@author: Varela
'''
import numpy as np 

def travel_builder(D, S, U, t, nteams):
    # Builds a matrix for maximum flow algorithm     
#     origin      = U[:,0,-1]-1 # From last UMPIRE LOCATIONS
    nmatches = int(nteams/2)
    origin      = U[t-1,:]   # From last UMPIRE LOCATIONS
    destination = S[t,:, 0]   # From next GAME LOCATIONS
    # from teams to index mappping 
    origin -=1
    destination -=1
    #index with all combinations
    R = np.tile(origin.reshape(nmatches,1),(1,nmatches))
    C = np.tile(destination.reshape(1,nmatches),(nmatches,1)) 
         
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

def constraint_violation_builder(D, S, U, t, d1, d2):
    # Produces a Cmask[numpires, ntemns]
    numpires = S.shape[1]
    # C4, C5 indicator
    Cmask     = np.zeros((numpires,2*numpires),dtype=bool)
    idumpires = np.arange(numpires).reshape(1,numpires)
    
    if t > 0:                
        y   =  max(t-(numpires-d1),0)  
        U4  = U[y:t,:]       # umpire game history
        S4  = S[y:t,:,0]     # last game venue
        L4  = S4[:,U4[0,:]-1]       # 
        colsindex4 = L4 - 1         # locations positions
#         C4  = D[IL4]
        rowsindex4 = np.tile(idumpires,(t-y,1))    
        Cmask[rowsindex4,colsindex4] = True
                             
        
#         y    =  max(t-int(numpires/2),0)  
#         U5   = U[y:t,:]       # umpire game history
#         SH5  = S[y:t,:,0]     # last game home team
#         SV5  = S[y:t,:,1]     # last game visit team
            
    return Cmask     
            
            
         
    