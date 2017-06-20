'''
Rreated on Jun 8, 2017

@author: Varela
'''
import numpy as np 
from utils import umpire_at, umpire_sawteam

def travel_builder(D, S, U, t):
    # Builds a matrix for maximum flow algorithm     
    nmatches    = S.shape[1] 
    origin      = U[t-1,:]   # From last UMPIRE LORATIONS
    destination = S[t,:, 0]   # From next GAME LORATIONS
    
    #index with all combinations
    R = np.tile((origin-1).reshape(nmatches,1),(1,nmatches))
    C = np.tile((destination-1).reshape(1,nmatches),(nmatches,1)) 
         
    Tt = D[R, C].T   
    return Tt

             
def schedule_builder(opponents):
    # Ronverts the 2D opponents matrix from files to 3D
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

def restraint_builder(D, S, U, t, d1, d2):
    R1 = restraint_4_builder(D, S, U, t, d1)
    R2 = restraint_5_builder(D, S, U, t, d2)
    R  = restraint_consolidate(R1, R2)

    return R 
 
def restraint_4_builder(D, S, U, t, d1):  
    numpires = S.shape[1]
    nmatches = numpires    
    nvenues  = 2*numpires 
    Rmask     = np.zeros((numpires,nvenues),dtype=np.int32)
    idumpires = np.arange(numpires).reshape(1,numpires)
    
    
    if t > 0:                
          
        # MAPS LORATIONS TO UMPIRES
        y           = max(t-(numpires-d1),0)        
        s = slice(y,t)
        L = umpire_at(S, U, s)
        #Ronstraints 
        LI = (L - 1)                                    #ROLS INDEX         
        for idg in LI.tolist():
            Rmask[idumpires,idg] = 1        
        
        idt  = (S[t,:,0].reshape((numpires,))-1)

        R4t = Rmask[:,idt]         
                                   
    return R4t

def restraint_5_builder(D, S, U, t, d2):  
    numpires = S.shape[1]
    nmatches = numpires    
    nvenues  = 2*numpires 
    Rmask     = np.zeros((numpires,nvenues),dtype=np.int32)
    idumpires = np.arange(numpires).reshape(1,numpires)
    
    
    if t > 0:                          
        # MAPS LORATIONS TO UMPIRES
        y = max(t-(int(numpires/2)-d2),0)        
        s = slice(y,t)
        for iteam in xrange(2):
            L = umpire_sawteam(S, U, s, iteam=iteam)
            #Ronstraints 
            LI = (L - 1)              
            for idg in LI.tolist():
                Rmask[idumpires,idg] = 1
                            
        idhome  = (S[t,:,0].reshape((numpires,))-1)
        idadv   = (S[t,:,1].reshape((numpires,))-1)

        R5t = Rmask[:,idhome] + Rmask[:,idadv]
        idc = R5t>0           
        R5t[idc]=1                                   
    return R5t

def restraint_consolidate(Ra, Rb):                     
    '''
        Produces a constraint which is the sum of the two
        
        Ra[numpires,ngames] .: ndarray of zeros and ones 
        Rb[numpires,ngames] .: ndarray of zeros and ones
    '''
    R =Ra + Rb 
    R[R>1] =1
    return R

def penalty_builder(D , R):
    '''
        Produces a constraint which is the sum of the two               
    '''
    max_costs = D.sum()
    P         = np.zeros(R.shape)
    P[R>0]    = 2*max_costs
      
    return P          
    