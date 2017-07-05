'''
Created on Jun 12, 2017

@author: Varela
'''

import numpy as np
import pandas as pd  
import writers as wr 
from umps import *
import scipy.optimize as opt 
from scipy.sparse import csr_matrix 
import networkx as nx 
from singletons import get_dictionary 


class TUP(object):
  '''
  TUP solution
  '''
  def __init__(self, D, S, q1, q2, fixpenalty):
  #def __init__(self):
    #Initialize 
    # dct = get_dictionary()
    # D  = dct['D']
    # S  = dct['S']
    # q1 = dct['q1'] 
    # q2 = dct['q2'] 
    # fixpenalty = dct['fixpenalty'] 

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
    '''
    Performs optimized local search using hungarian assignment 
    '''
    nrounds, numps, _ = S.shape 
    t = np.random.randint(1,nrounds)
    fixpenalty = self.fixpenalty
    # UX is the cartesian product
    UX  = umps2cartesian(self.U[:t,:], tup.U[t:,:])
    VX3 = umps2violations3(S, UX)
    VX4 = umps2violations4(S, UX, q1)
    VX5 = umps2violations5(S, UX, q2)
    TX  = umps2travel(D, S, UX)
    PX  = (VX3 + VX4 + VX5) * fixpenalty

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

  def sa(self, D, S, q1, q2, T0, verbose=True):
    '''
    Simulated annealing
    '''
    
    def saperm(x, V):
      '''
      Applies a permutation operator the time preceding a violation and applies a random permutation
      ''' 
      # Finds a violated i to operate
      nrounds, numps = x.shape
      index  = np.arange(1, nrounds+1)
      viol   = V.sum(axis=1).astype('bool')
      if viol.any(): 
        status = 0 
        r      = np.random.choice(index[viol],size=1)-1

        # Applies a full permutation operator on t
        index  = np.arange(1,numps+1)
        perm   = np.random.choice(index,size=numps,replace=False)
        x[r,index-1] = perm
      
        V = umps2violations3(S, x) 
        V += umps2violations4(S, x, q1)
        V += umps2violations5(S, x, q2)        
      else: 
        status = 1

      return x, V, status  

    w = self.U       
    Ik = 10  # repetition schedule for k
    i = 0 

    K = self.V3 + self.V4 + self.V5     
    T0 = int(T0)
    Tk = int(0.2*T0)
    T = np.linspace(T0,Tk)

    status  = 0 if K.any() else 1
    if status == 0: 
      for tk in T:
        for i in xrange(Ik):
          wprime, Kprime, stopcriteria = saperm(w,K)      

          if stopcriteria == 0: # violations present 
            delta = Kprime.sum() - K.sum()    
            if delta <= 0:
              w = wprime
              K = Kprime               

            elif (np.random.rand() < np.exp(-(delta/tk))):
              w = wprime
              K = Kprime               

                
          else:
            break 
        if stopcriteria==1:
          if verbose:
            print 'Stop criteria reached exiting GASA'
          break 

    self.U = w
    self.V3 = umps2violations3(S, w)
    self.V4 = umps2violations4(S, w, q1)
    self.V5 = umps2violations5(S, w, q2)
    self.T  = umps2travel(D, S, w)
    self.P  = (self.V3 + self.V4 + self.V5) * self.fixpenalty                  
    
  def repair(self, D, S, q1, q2, verbose=True):
    '''
    Applies hopcroft_karp_matching algorithm
    '''
    U  = self.U    
    V4  = self.V4 
    V5  = self.V5 
    repairindex = (V4+V5).sum(axis=1)
    if repairindex.any():
      # Stores the indexes of the forbidden teams
      
      nteams = D.shape[0]
      numps  = int(nteams/2)

      idt = np.arange(S.shape[0])
      idt = idt[repairindex>0]
      t = np.random.choice(idt,size=1)[0]  # has to be randomly drawn
      

      idu = np.arange(0,numps).reshape(1,numps)
      #FORBBIDEN BY VIOLATION 4
      F4 = np.zeros((nteams, numps))  
      Hom = umps2home(S,U)
      t0 = max(t-q1+1,0)
      
      idx = Hom[t0:t,:]-1 
      idy = np.tile(idu, (t-t0,1))
      
      F4[idx, idy] = 1

      #FORBBIDEN BY VIOLATION 5
      F5 = np.zeros((nteams, numps))
      t0 = max(t-q2+1,0)
      idx = Hom[t0:t,:]-1 
      idy = np.tile(idu, (t-t0,1))
      F5[idx, idy] = 1  

      Adv = umps2adversaries(S,U)
      idx = Adv[t0:t,:]-1
      F5[idx, idy] = 1  

      
      # CONVERTS FORBIDDEN HISTORY INTO FORBBIDEN FOR TIME T
      idx = np.tile(Hom[t,:]-1, (numps,1)).T 
      idy = np.tile(idu, (numps,1))
      FT4 = F4[idx,idy]
      
      
      FT5 = F5[idx,idy]
      idx = np.tile(Adv[t,:]-1, (numps,1)).T 
      FT5 += F5[idx,idy]

      A = 1 - FT4 - FT5 
      A[A<0] = 0 

      #MAXIMUM CARDINALITY HEURISTIC 
      K = csr_matrix(A)
      # Game vertices are shifted by numps
      G = nx.bipartite.from_biadjacency_matrix(K)
      dct = nx.bipartite.hopcroft_karp_matching(G)

      if verbose: 
        nviolprev= (V4[t,:] + V5[t,:]).any().sum()
        nviolnew = numps - int(len(dct.keys())/2)
        print "number of violations %d -> %d" %(nviolprev, nviolnew)
        

      # Defines new allocation - filling the positions 
      # in case no viable answer was found 
      idg = np.zeros((numps,), dtype=np.int32)
      masku = np.ones((numps,), dtype=bool)
      maskg = np.ones((numps,), dtype=bool)


      
      for u, g in dct.items():
        if u < numps:
       
          idg[u] = int(g - numps) # We have to adjust after transformation
          masku[u] = False 
          maskg[g-numps]= False
      
      idg[maskg] = U[t,masku]-1   

      idu = idu.reshape((numps,))

      # fills left overs in a sequential way    
      self.U[t,idu] = idg+1   
      self.V3 = umps2violations3(S, self.U)
      self.V4 = umps2violations4(S, self.U, q1)
      self.V5 = umps2violations5(S, self.U, q2)
      self.T  = umps2travel(D, S, self.U)
      self.P  = (self.V3 + self.V4 + self.V5) * self.fixpenalty                  


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

  def violations(self):
      return (self.V3 + self.V4 + self.V5).sum()     

  def persist(self, D, S, epochs, q1, q2, instancename, timestamp):    
    tupdf = self.to_frame(D,S)
    wr.persist_solution(tupdf, D, S, epochs, q1, q2, instancename, timestamp)
  
  def export1(self, instancename, timestamp, q1, q2):
    wr.persist_siteformat1(self._to_dfsite1(), instancename, timestamp, q1, q2)
      

  def export2(self, S,  instancename, timestamp, q1, q2):    
    wr.persist_siteformat2(self._to_dfsite2(S), instancename, timestamp, q1, q2)
  
  



  def _to_dfsite1(self):        
    nrounds, numps  =self.U.shape 
    G               = umps2games(self.U)
    expdata         = G.flatten()
    l               = nrounds*numps
    return pd.DataFrame(data=expdata.reshape((1,l)))           

  def _to_dfsite2(self, S):        
    exportdata      = umps2home(S, self.U)              
    return pd.DataFrame(data=exportdata)           
          
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

