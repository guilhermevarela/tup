'''
Created on Jun 8, 2017

@author: Varela

'''
import numpy as np
from utils import violations_3_counter
import scipy.optimize as opt

from builders import restraint_builder, restraint_4_builder, restraint_5_builder, travel_builder, restraint_segment_builder, penaltyfix_builder

class RandomNaiveMatchingSolver: 
    def __init__(self, D, S, d1, d2, fixpenalty=300):
        self.D = D 
        self.S = S 
        self.d1 =d1 
        self.d2 =d2         
        self.fixpenalty = fixpenalty
    
    def solve(self):
        D          = self.D  
        S          = self.S  
        d1         = self.d1 
        d2         = self.d2         
        fixpenalty = self.fixpenalty  
        
        # initialize umpires and violations
        nrounds, numpires, _ = S.shape
        #U, V3, V4, V5, T, P 
        U  = np.zeros((nrounds,numpires),dtype=np.int32)                
        V3 = np.zeros((nrounds,numpires),dtype=np.int32)
        V4 = np.zeros((nrounds,numpires),dtype=np.int32)
        V5 = np.zeros((nrounds,numpires),dtype=np.int32)
        T  = np.zeros((nrounds,numpires),dtype=np.int32)
        P  = np.zeros((nrounds,numpires),dtype=np.int32)

        # P = np.zeros((nrounds,numpires),dtype=np.int32)        
        # T     = np.zeros((nrounds,numpires),dtype=np.int32)

        umpiresindex = np.arange(numpires)
        for t in xrange(nrounds):
            if t == 0:
                U[0,:] = np.arange(numpires)+1 
                np.random.shuffle(U[0,:])
                t +=1
            else:                                                                  
                R4t = restraint_4_builder(S, U, t, d1)
                R5t = restraint_5_builder(S, U, t, d2)
                Tt = travel_builder(D, S, U, t)
                
                gamesindex   =  np.random.choice(umpiresindex,replace=False,size=numpires)
                                            

                U[t, umpiresindex] = (gamesindex+1)
                V4[t,:] = R4t[umpiresindex,gamesindex]
                V5[t,:] = R5t[umpiresindex,gamesindex]                
                T[t,:] = Tt[umpiresindex, gamesindex]
                P[t,:] = fixpenalty*(V[t,:])
                                                

        V3[-1,:] = violations_3_counter(S,U)        
        P[-1,:] += fixpenalty * V3[-1,:]
        #U, V3, V4, V5, T, P  = 
        #return U, V, T, P 
        return U, V3, V4, V5, T, P 
             
                                                                                                
#Implementation of the hungarian method         
class BipartiteMatchingSolver:
    def __init__(self, C):
        self.C = C # residual graph
    
    # Ia, Ij, c = solve()
    # Ia .: index of applicants
    # Ij .: index of jobs
    # c .: total T
    def solve(self):             
        Ia, Ij = opt.linear_sum_assignment(self.C)
        c = self.C[Ia,Ij].sum()
        return Ia, Ij, c

class BipartiteMatchingSolverR:
    def __init__(self, D, S, U, t, d1, d2, fixpenalty):
        self.D = D          
        self.S = S 
        self.U = U 
        self.t = t           
        self.d1 = d1 
        self.d2 = d2
        self.fixpenalty = fixpenalty
    
    
    def solve(self):
        D          = self.D           
        S          = self.S 
        U          = self.U 
        t          = self.t 
        d1         = self.d1  
        d2         = self.d2
        fixpenalty = self.fixpenalty        

        R = restraint_segment_builder(S, U, t, d1, d2)
        P = penaltyfix_builder(R, fixpenalty=fixpenalty)                       
        umpsindex, gamsindex = opt.linear_sum_assignment(D + P)
        return umpsindex, gamsindex
