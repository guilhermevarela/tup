'''
Created on Jun 12, 2017

@author: Varela
'''

import numpy as np
import pandas as pd  
import os 
 
# from solvers import RandomNaiveMatchingSolver, BipartiteMatchingSolverR
from builders import travel_builder
from utils import umpire_at, umpire2game, umpire2homevenue
from umps import *
import scipy.optimize as opt 


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
            U[t,:] = np.random.choice(umpsindex, size=numps, replace=False)+1

        self.U = U    
        self.V3 = umps2violations3(S, U)
        self.V4 = umps2violations4(S, U, d1)
        self.V5 = umps2violations5(S, U, d2)
        self.T = umps2travel(D, S, U)
        self.P = self.T * fixpenalty
        self.fixpenalty = fixpenalty

    def x(self, tup, D, S, d1, d2):
        nrounds, numps, _ = S.shape 
        t = np.random.randint(1,nrounds)
        fixpenalty = self.fixpenalty
        # UX is the cartesian product
        UX = umps2cartesian(self.U[:t,:], tup.U[t:,:])
        VX3 = umps2violations3(S, UX)
        VX4 = umps2violations4(S, UX, d1)
        VX5 = umps2violations5(S, UX, d2)
        TX = umps2travel(D, S, UX)
        PX = TX * fixpenalty

        # COMPUTES TXt, PXt for cut
        TXt = np.hsplit(TX.sum(axis=0),numps)
        PXt = np.hsplit(PX.sum(axis=0),numps)

        print TXt 
        print PXt 

        # Hungarian algorithm
        idumps, idgame = opt.linear_sum_assignment(TXt + PXt)
        
        #adjust to cross cartesian
        idgamex =  idgame + np.arange(0,numps*numps, numps)
        self.U  = UX[:,idgamex]    
        self.V3 = VX3[:,idgamex]
        self.V4 = VX4[:,idgamex]
        self.V5 = VX5[:,idgamex]
        self.T = TX[:,idgamex]
        self.P = PX[:,idgamex]
        
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