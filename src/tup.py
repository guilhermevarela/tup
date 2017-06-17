'''
Created on Jun 12, 2017

@author: Varela
'''

import numpy as np
import pandas as pd  
 
from solvers import RandomGreedyMatchingSolver, BipartiteMatchingSolver
from builders import travel_builder
class TUP(object):
    '''
    TUP solution stores d1, d2
    '''


    def __init__(self, D, S, d1, d2):                         
        cost, solution, violations =  RandomGreedyMatchingSolver(D, S, d1, d2).solve()
        nrounds             = S.shape[0]
        self.cost           = cost 
        self.score          = np.sum(cost)
        self.U              = solution 
        self.violations     = violations 
        self.nrounds        = nrounds
        
    def x(self, other_tupsolution, D, S, d1, d2):
        nrounds = self.nrounds
        t = np.random.randint(1,nrounds-1)
        
        #Settings before optimization        
        self.cost[t+1:]         = other_tupsolution.cost[t+1:] 
        self.U[t+1:,:]          = other_tupsolution.U[t+1:,:]
        self.violations[t+1:,:] = other_tupsolution.violations[t+1:,:]
        
        U = self.U
        c = self.cost 
        for t1 in xrange(t+1,nrounds):
            Tt  = travel_builder(D,S,U,t1)
            solver = BipartiteMatchingSolver(Tt)
            UI, GI, ct = solver.solve()
            U[t1,UI] = (GI+1)
            c[t1] = ct 

        self.cost = c 
        self.U = U 
        self.score = np.sum(c)
        return self

    def to_frame(self, D, S):
        '''
            to_frame generates a pandas dataframe
            
        '''
        nrounds, numpires, _ = S.shape
        c                    = np.cumsum( self.cost )
        umpirecolumns      = ['Umpire #%d'%(x+1) for x in xrange(numpires)]
        costscolumns       = ['D']
        index              =xrange(nrounds)
        
        permutationindex = (self.U-1)        
        Uout    = np.empty((nrounds,numpires), dtype=object)
        cout    = np.empty((nrounds,1), dtype=object)
         
        for r  in xrange(nrounds):
            p = permutationindex[r] 
            roundlist = S[r][p]
            cout[r] = "{:,}".format(c[r]) 
            for t, tuplelist in enumerate(roundlist):
                Uout[r,t]  =  '(%02d,%02d)' % tuple(tuplelist)
                 
        columns = umpirecolumns + costscolumns
        outdata = np.concatenate((Uout, cout),axis=1)
        return pd.DataFrame(data=outdata, columns=columns, index=index)
        
        