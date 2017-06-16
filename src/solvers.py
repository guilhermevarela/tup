'''
Created on Jun 8, 2017

@author: Varela

'''
import numpy as np
import scipy.optimize as opt

from builders import constraint_4_builder, travel_builder

class ProbabilisticGreedyMatchingSolver:    
    def __init__(self, D, S, d1, d2):
        self.D = D 
        self.S = S 
        self.d1 =d1 
        self.d2 =d2 
        
        
    def solve(self):
        D = self.D  
        S = self.S          
        d1 = self.d1
                 
        
        # initialize umpires and violations
        nrounds, numpires, _ = S.shape
        U = np.zeros((nrounds,numpires),dtype=np.int32)                
        V = np.zeros((nrounds,numpires),dtype=np.int32)
        c = np.zeros((nrounds,),dtype=np.int32)
        # random assignment at time 0
        U[0,:] = np.arange(numpires) +1
        np.random.shuffle(U[0,:])
        
        for t in xrange(1,nrounds):

            Ct = constraint_4_builder(D, S, U, t, d1)
            Tt = travel_builder(D, S, U, t)
            
            # Greedy Heuristic -> Take the shortest distance possible, using stable marriage
            #uindex, gindex = StableMatchingSolver(Tt).solve()
            uindex, gindex = StableMatchingSolverR(Tt, Ct).solve()
            
            U[t, uindex] = (gindex+1)
            c[t] = Tt[uindex, gindex].sum()
        return c, U, V
                    
                
                                                
                 
class StableMatchingSolver:
    '''
        Conventional StableMatchingSolver input is a cost matrix - both umpires and games prefer the 
        minimum cost.
    '''
    def __init__(self, C):
        self.C = C
        self.UP =  np.argsort(C, axis=1)
        self.GP =  np.argsort(C, axis=0)
                 
        
    def solve(self):
        UP  =self.UP
        GP  =self.GP
        
        
        numpires, ngames = UP.shape
        umpirefree = np.ones((numpires,), dtype=bool)
        gamefree   = np.ones((ngames,)  , dtype=bool)
            
        indexumpires = np.arange(numpires, dtype=np.int32)
        indexgames   = np.zeros((ngames,), dtype=np.int32) 
            
        # Saves the index of the last proposition - "engagement"
        engagement = np.zeros((numpires, ),dtype=np.int32)             
        while (umpirefree.any() | gamefree.any()):
            unmatchedumpires =  indexumpires[umpirefree]
                                    
            for u in unmatchedumpires:
                g = UP[u,engagement[u]]
                if gamefree[g]: 
                    indexgames[g] = u
                    
                    gamefree[g]   = False 
                    umpirefree[u] = False 
                else: 
                    prevu =indexgames[g]
                    # least index greater the preference
                    prefu     = np.where( GP[:,g] ==u )[0][0]
                    prefprevu = np.where( GP[:,g] ==prevu )[0][0] 
                    if prefu  < prefprevu:
                        indexgames[g] = u
                        umpirefree[u] = False
                        umpirefree[prevu] = True 
                        engagement[prevu] +=1
                    else:                          
                        engagement[u] +=1    
                              
        return indexumpires, indexgames                

class StableMatchingSolverR:
    '''
        StableMatchingSolver with restrictions - some pairings are forbidden  
    '''
    def __init__(self, C, R):
        self.C  =  C
        self.UP =  np.argsort(C, axis=1)
        self.GP =  np.argsort(C, axis=0)
        self.R  = R
    
    def prefersu2v(self, P, u, v):    
        '''
            Compares canditates u,v using preference array P - of canditates, lower the index
            greater the preference
            
            returns u if u=argmax(P | u,v) esle v=argmax(P | u,v)
        '''
        prefu     = np.where( P ==u )[0][0]
        prefv     = np.where( P ==v )[0][0]
        return prefu if prefu < prefv else prefv
    
    def prefers(self, P, u):    
        '''
            Compares array of canditates u using preference array P - of canditates, lower the index
            greater the preference
            
            returns argmax(P | u) 
        '''
        l = len(P)
        S = np.array(xrange(l,0,-1))
        i = np.argmax(S[P[u]])        
        return u[i]
                  
    def solve(self):
        UP  =self.UP
        GP  =self.GP
        R   =self.R
        
        numpires, ngames = UP.shape
        umpirefree = np.ones((numpires,), dtype=bool)
        gamefree   = np.ones((ngames,)  , dtype=bool)
            
        indexumpires = np.arange(numpires, dtype=np.int32)
        indexgames   = np.zeros((ngames,), dtype=np.int32) 
            
        # Saves the index of the last proposition - "engagement"
        engagement = np.zeros((numpires, ),dtype=np.int32)             
        while (umpirefree.any() | gamefree.any()):
            unmatchedumpires =  indexumpires[umpirefree]
                                    
            for u in unmatchedumpires:
                if engagement[u] < ngames: 
                    g = UP[u,engagement[u]]
                    forbidden =  bool(R[u,engagement[u]])                
                    if forbidden >0:
                        engagement[u] +=1
                    elif gamefree[g]: 
                        indexgames[g] = u
                    
                        gamefree[g]   = False 
                        umpirefree[u] = False 
                    else: 
                        prevu =indexgames[g]
                        # least index greater the preference
#                         prefu     = np.where( GP[:,g] ==u )[0][0]
#                         prefprevu = np.where( GP[:,g] ==prevu )[0][0]
                        pref =  self.prefersu2v(GP[:,g], u, prevu) 
                        if u == pref:
                            indexgames[g] = u
                            umpirefree[u] = False
                            umpirefree[prevu] = True 
                            engagement[prevu] +=1
                        else:                          
                            engagement[u] +=1
                else: 
                    # from all free games returns the index of the favorite                    
                    bestscore=ngames 
                    bestg    = -1
                    for g in xrange(ngames):
                        if gamefree[g]: 
                            score = np.where(GP[:, g] == u)[0][0]
                            if score<bestscore:
                                bestscore=score
                                bestg=g 
                                 
                    indexgames[bestg] = u
    
                    gamefree[bestg]   = False 
                    umpirefree[u] = False
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
        