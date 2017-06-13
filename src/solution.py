'''
Created on Jun 12, 2017

@author: Varela
'''
import numpy as np 
from solvers import ProbabilisticGreedyMatchingSolver, BipartiteMatchingSolver
from builders import travel_builder
class TUPSolution(object):
    '''
    TUP solution stores d1, d2
    '''


    def __init__(self, D, S, d1, d2):                         
        cost, solution, violations =  ProbabilisticGreedyMatchingSolver(D, S, d1, d2).solve()
        nrounds             = S.shape[0]
        self.cost           = cost 
        self.score          = np.sum(cost)
        self.solution       = solution 
        self.violations     = violations 
        self.nrounds        = nrounds
        
    def x(self, other_tupsolution, D, S, d1, d2):
        nrounds = self.nrounds
        t = np.random.randint(1,nrounds-1)
        
        #Settings before optimization        
        self.cost[t+1:]         = other_tupsolution.cost[t+1:] 
        self.solution[t+1:,:]   = other_tupsolution.solution[t+1:,:]
        self.violations[t+1:,:] = other_tupsolution.violations[t+1:,:]
        
        U = self.solution
        c = self.cost 
        for t1 in xrange(t+1,nrounds):
            Tt  = travel_builder(D,S,U,t1)
            solver = BipartiteMatchingSolver(Tt)
            UI, GI, ct = solver.solve()
            U[t1,UI] = GI
            c[t1] = ct 

        self.cost = c 
        self.solution = U 
        self.score = np.sum(c)
        return self

    def to_frame(self):
        '''
            to_frame generates a pandas dataframe
        '''
        raise NotImplementedError('to_frame not implemented yet')