'''
Rreated on Jun 8, 2017

@author: Varela
'''
import numpy as np 
from utils import umpire_at, umpire_sawteam, umpire2segment

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

def restraint_builder(S, U, t, d1, d2):
    R4 = restraint_4_builder(S, U, t, d1)
    R5 = restraint_5_builder(S, U, t, d2)
    R  = restraint_consolidate(R4, R5)

    return R 
 
def restraint_4_builder(S, U, t, d1):  
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

def restraint_5_builder(S, U, t, d2):  
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

        R5t = (Rmask[:,idhome] + Rmask[:,idadv])
        
        idc = R5t>0           
        R5t[idc]=1                                   
    return R5t

def restraint_consolidate(Ra, Rb):                     
    '''
        Produces a constraint which is the sum of the two
        
        Ra[numpires,nteams] .: ndarray of zeros and ones 
        Rb[numpires,nteams] .: ndarray of zeros and ones
    '''
    R =Ra + Rb
     
    R[R>1] =1
    return R

def restraint_segment_builder(S, U, t, d1, d2):
    '''
        Produces a constraint which is the sum of the two
        R[numpires,nsegments,3] .: number of conflics for adopting
    '''    
    nrounds, numpires =U.shape
    R = np.zeros((numpires,numpires,3))
    

    segments = umpire2segment(S, U)
    for u in xrange(numpires):
        umpsegment = segments[u][:t]     
        for s in xrange(numpires):            
            newsegment = segments[s][t:]
            
            R[u,s,0] = restrain_3_bysegment(umpsegment, newsegment, 2*numpires)
            R[u,s,1] = restrain_4_bysegment(umpsegment, newsegment, numpires, d1) 
            R[u,s,2] = restrain_5_bysegment(umpsegment, newsegment, numpires, d2)
    
    return R

def restrain_3_bysegment(umpsegment, newsegment, nteams):
    # restraint 3 - all umpires must visit a team at home at least once
    teamcounter =  np.zeros((nteams,))
    
    for match in umpsegment:
        teamcounter[match[0]-1]+=1
    
    for match in newsegment:
        teamcounter[match[0]-1]+=1        

    return (teamcounter == 0).sum()     

def restrain_4_bysegment(umpsegment, newsegment, numpires, d1):
    # restraint 4 - no umpire is in a homesite more then once in any n-d1 consecutive slots
    nlookback        = min(numpires-d1,len(umpsegment))
    violations       = 0 
    if nlookback > 0:
        umpd1matches     = umpsegment[-nlookback:]
        umpd1homes       = list(zip(*umpd1matches)[0])
        newd1matches     = newsegment[:nlookback]
        newd1homes       = list(zip(*newd1matches)[0])               
    
        violations=len(set(umpd1homes).intersection(newd1homes))        
            
    return  violations

def restrain_5_bysegment(umpsegment, newsegment, numpires, d2):
    # restraint 5 - no umpire sees a team more then once in any int(n/2)-d2 consecutive slots
    nlookback = min(int(numpires/2)-d2, len(umpsegment))
    violations= 0
    if nlookback > 0:
        umpd2matches     = umpsegment[-nlookback:]
        umpd2homes       = list(zip(*umpd2matches)[0])
        umpd2adv         = list(zip(*umpd2matches)[1])
    
        newd2matches     = newsegment[:nlookback]
        newd2homes       = list(zip(*newd2matches)[0])
        newd2adv         = list(zip(*newd2matches)[1])               
    
        violations+=len(set(umpd2homes).intersection(newd2homes))
        violations+=len(set(umpd2homes).intersection(newd2adv))
        violations+=len(set(umpd2adv).intersection(newd2homes))
        violations+=len(set(umpd2adv).intersection(newd2adv))
                
    return  violations
                     

    
def penaltyfix_builder(R,fixpenalty=100):
    '''
        Produces a constraint which is the sum of the two               
    '''
    
    P         = R.sum(axis=2)*fixpenalty      
    return P          
    
    
    