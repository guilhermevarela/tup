'''
Created on Jun 12, 2017

@author: Varela
'''
import numpy as np 
from solvers import ProbabilisticGreedyMatchingSolver, BipartiteMatchingSolver

class TUPSolution(object):
    '''
    TUP solution stores d1, d2
    '''


    def __init__(self, D, S, d1, d2):                         
        cost, solution, violations =  ProbabilisticGreedyMatchingSolver(D, S, d1, d2).solve()
        nrounds = S.shape[0]
        self.cost = cost 
        self.solution = solution 
        self.violations = violations 
        self.nrounds = nrounds
        
    def x(self, other_tupsolution, D, S, d1, d2):
        nrounds = self.nrounds
        t = np.random.randint(1,nrounds-1)
        
        #Settings before optimization        
        self.cost[t+1:]        = other_tupsolution.cost[t+1,:] 
        self.solution[t+1:,:]   = other_tupsolution.solution[t+1:,:]
        self.violations[t+1:,:] = other_tupsolution.violations[t+1:,:]
        
        for t1 in xrange(t+1,nrounds):
        BipartiteMatchingSolver()
        
        return self