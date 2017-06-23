'''
Created on Jun 12, 2017

@author: Varela
'''

import numpy as np
import pandas as pd  
import os 
from umps import *
import scipy.optimize as opt 


class TUP(object):
    '''
    TUP solution
    '''
    def __init__(self, D, S, q1, q2, fixpenalty):
        #Defines variables 
        nrounds, numps, _ = S.shape 

        U  = np.zeros((nrounds,numps),dtype=np.int32)                        

        #constructs initial solution
        umpsindex = np.arange(numps)
        for t in xrange(nrounds):            
            U[t,:] = np.random.choice(umpsindex, size=numps, replace=False)+1

        self.U = U    
        
        self.V3 = umps2violations3(S, U)
        self.V4 = umps2violations4(S, U, q1)
        self.V5 = umps2violations5(S, U, q2)        
        

        self.T = umps2travel(D, S, U)
        self.P = (self.V3 + self.V4 + self.V5) * fixpenalty
        self.fixpenalty = fixpenalty

    def x(self, tup, D, S, q1, q2):
        nrounds, numps, _ = S.shape 
        t = np.random.randint(1,nrounds)
        fixpenalty = self.fixpenalty
        # UX is the cartesian product
        UX = umps2cartesian(self.U[:t,:], tup.U[t:,:])
        VX3 = umps2violations3(S, UX)
        VX4 = umps2violations4(S, UX, q1)
        VX5 = umps2violations5(S, UX, q2)
        TX = umps2travel(D, S, UX)
        PX = (VX3 + VX4 + VX5) * fixpenalty

        # COMPUTES TXt, PXt for cut
        TXt = TX[t,:].reshape((numps,numps))
        PXt = (PX[t:,:].sum(axis=0)).reshape((numps,numps))

        
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

    def mutate(self, D, S, q1, q2):
        U = self.U
        fixpenalty = self.fixpenalty


        nrounds, numps = U.shape
        tmutation = np.random.choice(np.arange(nrounds), size=1) 
        ujmutation = np.random.choice(np.arange(numps), size=2, replace=False) 
        uimutation = np.roll(ujmutation, 1)

        
        U[tmutation,uimutation] = U[tmutation,ujmutation]
        
        self.V3 = umps2violations3(S, U)
        self.V4 = umps2violations4(S, U, q1)
        self.V5 = umps2violations5(S, U, q2)
        self.T  = umps2travel(D, S, U)
        self.P  = (self.V3 + self.V4 + self.V5) * fixpenalty

    def to_frame(self, D, S):
        '''
            to_frame generates a pandas dataframe
            
        '''
        nrounds, numps, _ = S.shape
        umpiresdf = self._umps2frame_(S, nrounds, numps)
        violationsdf = self._violations2frame_(S, nrounds, numps) 
        penaltiesdf = self._penalties2frame_(S, nrounds, numps)
        traveldf = self._travel2frame_(S, nrounds, numps) 
        

        df = pd.concat((umpiresdf, violationsdf, penaltiesdf,traveldf),axis=1)

        return df 

    def score(self):        
        return self.P.sum() + self.T.sum()

    def travel(self):    
        return self.T.sum()

    def persist(self,D, S, epochs, q1, q2, instancename, timestamp, ouput_dir=''):
        if not(ouput_dir):
            output_dir = "../src/output/%s/%d/" %(instancename,timestamp)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                
            
        df          = self.to_frame(D,S)
        ganame      = 'ga_%s_i%04d-q1_%02d-q2_%02d.csv' % (instancename,epochs,q1,q2)
        filepath    = output_dir + ganame  
        df.to_csv(filepath, sep=',')
    
    def export1(self, instancename, timestamp, q1, q2, ouput_dir=''):
        # Exports to format in 
        # https://benchmark.gent.cs.kuleuven.be/tup/en/my_submissions/
        if not(ouput_dir):
            output_dir = "../src/output/%s/%d/" %(instancename,timestamp)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

        U               = self.U
        nrounds, numps     =U.shape 
        exportname      = '%s_%d_%d.txt' % (instancename,q1,q2)        
        filepath        = output_dir + exportname 
        G               = umpire2game(self.U)
        expdata         = G.flatten()
        l = nrounds*numps
        exportdf        = pd.DataFrame(data=expdata.reshape((1,l)))           
        exportdf.to_csv(filepath, sep=',', header=False, index=False)                

    def export2(self, S,  instancename, timestamp, q1, q2, ouput_dir=''):
        # Exports to format in 
        # https://benchmark.gent.cs.kuleuven.be/tup/en/my_submissions/
        if not(ouput_dir):
            output_dir = "../src/output/%s/%d/" %(instancename,timestamp)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
        
        U = self.U
        _, numps=U.shape 
        exportname      = '%s_%d_%d.txt' % (instancename,q1, q2)        
        filepath        = output_dir + exportname 
        exportdata      = umps2home(S, U)        
        exportdf        = pd.DataFrame(data=exportdata)           
        exportdf.to_csv(filepath, sep=' ', header=False, index=False, encoding='utf-8')                

    
    
    
    def _umps2frame_(self, S, nrounds, numps):                
        umpirecolumns      = ['Umpire#%d'%(x+1) for x in xrange(numps)]        
        index              =xrange(nrounds)        
        UI = np.array( self.U )-1 

        Uout    = np.empty((nrounds,numps), dtype=object)                 
        for r  in xrange(nrounds):
            roundlist = S[r, UI[r,:]]
            for t, tuplelist in enumerate(roundlist):
                Uout[r,t]  =  '(%02d,%02d)' % tuple(tuplelist)

        return pd.DataFrame(data=Uout,columns=umpirecolumns,index=index)

    def _violations2frame_(self, S, nrounds, numps):                
        V = np.concatenate((self.V3,self.V4,self.V5),axis=1)
        vcolumns = [] 
        for v in xrange(3,6,1):
            for x in xrange(numps):
                vcolumns.append('V%d#%d'%(v,x+1)) 
        index    =xrange(nrounds)        
        return pd.DataFrame(data=V,columns=vcolumns,index=index)

    def _penalties2frame_(self, S, nrounds, numps):                
        P = (self.V3 + self.V4 + self.V5)*self.fixpenalty

        pcolumns = ['P#%d'%(x+1) for x in xrange(numps)]                
        index    =xrange(nrounds)        
        return pd.DataFrame(data=P,columns=pcolumns,index=index)        

    def _travel2frame_(self, S, nrounds, numps):                
        T = self.T
        tcolumns = ['T#%d'%(x+1) for x in xrange(numps)]                
        index    =xrange(nrounds)        
        
        return pd.DataFrame(data=T,columns=tcolumns,index=index)        

