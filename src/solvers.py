'''
Created on Jun 8, 2017

@author: Varela

'''
import numpy as np
import scipy.optimize as opt

from builders import *

class ProbabilisticGreedyMatchingSolver:    
    def __init__(self, D, S, d1, d2):
        self.D = D 
        self.S = S
        self.d1 =d1 
        self.d2 =d2 
        
        
    def solve(self):
        D = self.D  
        S = self.S 
#         d1 = self.d1
#         d2 = self.d2
                 
        
        # initialize umpires and violations
        nrounds, numpires, _ = S.shape
        U = np.zeros((nrounds,numpires),dtype=np.int32)                
        V = np.zeros((nrounds,numpires),dtype=np.int32)
        c = np.zeros((nrounds,),dtype=np.int32)
        # random assignment at time 0
        U[0,:] = np.arange(numpires) +1
        np.random.shuffle(U[0,:])
        
        for t in xrange(1,nrounds):
#             Cmask        = constraint_violationmask_builder(D, S, U, t, d1, d2)
#             indexhome    = S[t,:,0]-1
#             Ct           = Cmask[:,indexhome]                        
                          
            
            Tt = travel_builder(D, S, U, t)
            
            # Greedy Heuristic -> Take the shortest distance possible, using stable marriage
            uindex, gindex = StableMatchingSolver(Tt).solve()
            
            U[t, uindex] = (gindex+1)
            c[t] = Tt[uindex, gindex].sum()
        return c, U, V
                    
                
                        
            
                        
                 
class StableMatchingSolver:
    def __init__(self, C):
        self.C = C 
    
    def solve(self):
        C = self.C
        
        numpires, ngames = C.shape
        umpirepref =  np.argsort(C, axis=1)
        gamepref   =  np.argsort(C, axis=0)  
            
        umpirefree = np.ones((numpires,), dtype=bool)
        gamefree   = np.ones((ngames,), dtype=bool)
            
        indexumpires = np.arange(numpires, dtype=np.int32)
        indexgames   = np.zeros((ngames,), dtype=np.int32) 
            
        # last proposition
        Pu = np.zeros((numpires, ),dtype=np.int32)             
        while (umpirefree.any() | gamefree.any()):
            unmatchedumpires =  indexumpires[umpirefree]
                                    
            for u in unmatchedumpires:
                g = umpirepref[u,Pu[u]]
                if gamefree[g]: 
                    indexgames[g] = u
                    
                    gamefree[g]   = False 
                    umpirefree[u] = False 
                else: 
                    prevu =indexgames[g]
                    # least index greater the preference
                    prefu     = np.where( gamepref[:,g] ==u )[0][0]
                    prefprevu = np.where( gamepref[:,g] ==prevu )[0][0] 
                    if prefu  < prefprevu:
                        indexgames[g] = u
                        umpirefree[u] = False
                        umpirefree[prevu] = True 
                        Pu[prevu] +=1
                    else:                          
                        Pu[u] +=1    
                              
        return indexumpires, indexgames                
     
#Implementation of the hungarian method         
class BipartiteMatchingSolver:
    def __init__(self, C):
        self.C = C # residual graph
    
    # Ia, Ij, c = solve()
    # Ia .: index of applicants
    # Ij .: index of jobs
    # c .: total costs
    def solve(self):             
        Ia, Ij = opt.linear_sum_assignment(self.C)
        c = self.C[Ia,Ij].sum()
        return Ia, Ij, c
        