'''
Created on Jun 12, 2017

@author: Varela
'''

import numpy as np
import pandas as pd  
import signal 
from solvers import RandomGreedyMatchingSolver, BipartiteMatchingSolverR
from builders import travel_builder, restraint_builder
from utils import umpire_at
class TUP(object):
    '''
    TUP solution stores d1, d2
    '''


    def __init__(self, D, S, d1, d2, fixpenalty):

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
            solver = BipartiteMatchingSolverR(Tt, S, U, t1, d1, d2)
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
        umpiresdf    = self._umpires_to_frame_(S, nrounds, numpires)
        homedf       = self._home_to_frame(S, nrounds, numpires) 
        costsdf      = self._costs_to_frame_(nrounds, numpires) 

        df = pd.concat((umpiresdf, homedf, costsdf),axis=1)

        return df 

    def score(self):
        return self.cost.sum() + self.penalty.sum()    
    def _violations_to_frame_(self, nrounds, numpires):
        umpires        = xrange(numpires) 
        columns        = ['C[%d,%d]'%(x+1,y+1) for x in umpires for y in umpires]        
                
        violations = np.array(self.violations)        

        return pd.DataFrame(data=violations, columns=columns,index=xrange(nrounds))
    
    def _umpires_to_frame_(self, S, nrounds, numpires):                
        umpirecolumns      = ['Umpire#%d'%(x+1) for x in xrange(numpires)]        
        index              =xrange(nrounds)        
        UI = np.array( self.U )-1 

        Uout    = np.empty((nrounds,numpires), dtype=object)                 
        for r  in xrange(nrounds):
            roundlist = S[r, UI[r,:]]
            for t, tuplelist in enumerate(roundlist):
                Uout[r,t]  =  '(%02d,%02d)' % tuple(tuplelist)
    
        return pd.DataFrame(data=Uout, columns=umpirecolumns,index=xrange(nrounds))                     
    
    def _games_to_frame_(self,nrounds, numpires):
        gamescolumns      = ['Games #%d'%(x+1) for x in xrange(numpires)]        
        Gout              = np.zeros((nrounds, numpires), dtype=np.int32)
        for r  in xrange(nrounds):
            Gout[r,:] = self.U[r,:]

        return pd.DataFrame(data=Gout, columns=gamescolumns,index=xrange(nrounds))                     
    def _home_to_frame(self,S ,nrounds, numpires):
        home_cols        = ['Umpire#%d  @'%(x+1) for x in xrange(numpires)]                 
        home_data        = umpire_at(S, self.U, slice(0,nrounds))
        return pd.DataFrame(data=home_data, columns=home_cols,index=xrange(nrounds))
        
    def _costs_to_frame_(self, nrounds, numpires) :
        c                    = np.cumsum( self.cost )
        cout    = np.empty((nrounds,1), dtype=object)
        for r  in xrange(nrounds):
            cout[r] = "{:,}".format(c[r]) 
        columns       = ['D']        
        return pd.DataFrame(data=cout, columns=columns,index=xrange(nrounds))                     