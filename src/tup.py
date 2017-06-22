'''
Created on Jun 12, 2017

@author: Varela
'''

import numpy as np
import pandas as pd  
import os 
 
from solvers import RandomNaiveMatchingSolver, BipartiteMatchingSolverR
from builders import travel_builder
from utils import umpire_at, umpire2game, umpire2homevenue
from umps import * 

class TUP(object):
    '''
    TUP solution stores d1, d2
    '''


    def __init__(self, D, S, d1, d2, fixpenalty):
        #Defines variables 
        nrounds, numps, _ = S.shape 

        U  = np.zeros((nrounds,numps),dtype=np.int32)                        

        #constructs initial solution
        umpsindex = np.arange(numps)
        for t in xrange(nrounds):            
            U[t,:] = np.random.choice(umpsindex, size=numps, replace=False)
        
        self.V3 = umps2violations3(S, U)
        self.V4 = umps2violations4(S, U, d1)
        self.V5 = umps2violations5(S, U, d2)
        self.T = umps2travel(D, S, U)
        self.P = T * fixpenalty
        self.fixpenalty = fixpenalty

    def x(self, tup, D, S, d1, d2):
        nrounds = self.nrounds
        t = np.random.randint(1,nrounds-1)
        
        #Settings before optimization        
        self.costs[t+1:,:]         = tup.costs[t+1:,:] 
        self.U[t+1:,:]             = tup.U[t+1:,:]
        self.violations[t+1:,:]    = tup.violations[t+1:,:]
        self.penalties[t+1:,:]       = tup.penalties[t+1:,:]
         
        U = self.U
        c = self.costs 
        v = self.violations
        p = self.penalties
        fixpenalty = self.fixpenalty
        for t1 in xrange(t+1,nrounds):            
            Tt  = travel_builder(D,S,U,t1)
            solver = BipartiteMatchingSolverR(Tt, S, U, t1, d1, d2, fixpenalty)
            # return Ia, Ij, c, v ,p      
            UI, GI  = solver.solve()


        
        self.U = U 
        self.costs = c 
        self.violations = violations
        self.penalties = penalties
        return self

    def to_frame(self, D, S):
        '''
            to_frame generates a pandas dataframe
            
        '''
        nrounds, numps, _ = S.shape
        umpiresdf    = self._umpires_to_frame_(S, nrounds, numps)
        homedf       = self._home_to_frame(S, nrounds, numps) 
        costsdf      = self._costs_to_frame_(nrounds, numps) 

        df = pd.concat((umpiresdf, homedf, costsdf),axis=1)

        return df 

    def score(self):
        return self.costs.sum() + self.penalties.sum()
    
    def persist(self,D, S, epochs, d1, d2, instancename, timestamp, ouput_dir=''):
        if not(ouput_dir):
            output_dir = "../src/output/%s/%d/" %(instancename,timestamp)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                
            
        df          = self.to_frame(D,S)
        ganame      = 'ga_%s_i%04d-d1_%02d-d2_%02d.csv' % (instancename,epochs,d1,d2)
        filepath    = output_dir + ganame  
        df.to_csv(filepath, sep=',')
    
    def export1(self, instancename, timestamp, ouput_dir=''):
        # Exports to format in 
        # https://benchmark.gent.cs.kuleuven.be/tup/en/my_submissions/
        if not(ouput_dir):
            output_dir = "../src/output/%s/%d/" %(instancename,timestamp)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

        U               = self.U
        nrounds, numps     =U.shape 
        q1              = numps
        q2              = int(numps/2)        
        exportname      = '%s_%d_%d.txt' % (instancename,q1,q2)        
        filepath        = output_dir + exportname 
        G               = umpire2game(self.U)
        expdata         = G.flatten()
        l = nrounds*numps
        exportdf        = pd.DataFrame(data=expdata.reshape((1,l)))           
        exportdf.to_csv(filepath, sep=',', header=False, index=False)                

    def export2(self, S,  instancename, timestamp, ouput_dir=''):
        # Exports to format in 
        # https://benchmark.gent.cs.kuleuven.be/tup/en/my_submissions/
        if not(ouput_dir):
            output_dir = "../src/output/%s/%d/" %(instancename,timestamp)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
        
        U = self.U
        _, numps=U.shape 
        q1 = numps
        q2 = int(numps/2)
        exportname      = '%s_%d_%d.txt' % (instancename,q1, q2)        
        filepath        = output_dir + exportname 
        exportdata      = umpire2homevenue(S, U)        
        exportdf        = pd.DataFrame(data=exportdata)           
        exportdf.to_csv(filepath, sep=' ', header=False, index=False, encoding='utf-8')                

    def _violations_to_frame_(self, nrounds, numps):
        umpires        = xrange(numps) 
        columns        = ['C[%d,%d]'%(x+1,y+1) for x in umpires for y in umpires]        
                
        violations = np.array(self.violations)        

        return pd.DataFrame(data=violations, columns=columns,index=xrange(nrounds))
    
    def _umpires_to_frame_(self, S, nrounds, numps):                
        umpirecolumns      = ['Umpire#%d'%(x+1) for x in xrange(numps)]        
        index              =xrange(nrounds)        
        UI = np.array( self.U )-1 

        Uout    = np.empty((nrounds,numps), dtype=object)                 
        for r  in xrange(nrounds):
            roundlist = S[r, UI[r,:]]
            for t, tuplelist in enumerate(roundlist):
                Uout[r,t]  =  '(%02d,%02d)' % tuple(tuplelist)
    
        return pd.DataFrame(data=Uout, columns=umpirecolumns,index=xrange(nrounds))                     
    
    def _games_to_frame_(self,nrounds, numps):
        gamescolumns      = ['Games #%d'%(x+1) for x in xrange(numps)]        
        Gout              = np.zeros((nrounds, numps), dtype=np.int32)
        for r  in xrange(nrounds):
            Gout[r,:] = self.U[r,:]

        return pd.DataFrame(data=Gout, columns=gamescolumns,index=xrange(nrounds))                     
    def _home_to_frame(self,S ,nrounds, numps):
        home_cols        = ['Umpire#%d  @'%(x+1) for x in xrange(numps)]                 
        home_data        = umpire_at(S, self.U, slice(0,nrounds))
        return pd.DataFrame(data=home_data, columns=home_cols,index=xrange(nrounds))
        
    def _costs_to_frame_(self, nrounds, numps) :
        c                    = np.cumsum( self.costs )
        cout    = np.empty((nrounds,1), dtype=object)
        for r  in xrange(nrounds):
            cout[r] = "{:,}".format(c[r]) 
        columns       = ['D']        
        return pd.DataFrame(data=cout, columns=columns,index=xrange(nrounds))                     