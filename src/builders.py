'''
Created on Jun 8, 2017

@author: Varela
'''
import numpy as np 

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

def constraint_violationmask_builder(D, S, U, t, d1, d2):
    # Produces a Cmask[numpires, ntemns]
    numpires = S.shape[1]
    # C4, C5 indicator
    Cmask     = np.zeros((numpires,2*numpires),dtype=np.int32)
    idumpires = np.arange(numpires).reshape(1,numpires)
    
    if t > 0:                
        y   =  max(t-(numpires-d1),0)  
        U4  = U[y:t,:]                                  # umpire game history
        S4  = S[y:t,:,0].reshape((t-y,numpires))       # last game venue
        L4  = S4[:,U4[0,:]-1]        
        colsindex4 = L4 - 1                             # locations positions
        rowsindex4 = np.tile(idumpires,(t-y,1))    
        Cmask[rowsindex4,colsindex4] = 1
                             
            
    return Cmask     
            
def constraint_4_builder(D, S, U, t, d1):
    # C4[numpires, ngames] where 0 doesn't have a penalty  
    numpires = S.shape[1]
    nmatches = numpires    
    nvenues  = 2*numpires 
    Cmask     = np.zeros((numpires,nvenues),dtype=np.int32)
    idumpires = np.arange(numpires).reshape(1,numpires)
    
    
    if t > 0:                
        y           = max(t-(numpires-d1),0)  
        UI          = np.array( U[y:t,:]-1 )                                  
        HV          = S[y:t,:,0]                                #ALL HOME VENUES
#         L           = HV[:,UI[0]]                               #UMPIRE LOCATIONS AT t-1
        # MAPS LOCATIONS TO UMPIRES
        L = np.zeros(UI.shape, dtype=np.int32)
        for i, idx in enumerate(UI.tolist()):
            L[i,idx] = HV[i,:]
#         C           = L - 1                                     #COLS INDEX 
#         R           = np.tile(idumpires,(t-y,1))    
        #Constraints 
        LI = (L - 1)                                    #COLS INDEX         
        for idg in LI.tolist():
            Cmask[idumpires,idg] = 1
#         Cmask[R,C] = 1
        
        
        idt  = (S[t,:,0].reshape((numpires,))-1)
#         VIt  = (HVt-1)
        C4t = Cmask[:,idt] 
        print 'Track record Home venues'
        print L  
        print 'Current home venue',
        print S[t,:,0].reshape((numpires,))
        print 'Constraint'
        print C4t
                                   
    return C4t     
                
         
    