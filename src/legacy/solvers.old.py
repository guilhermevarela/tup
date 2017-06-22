class RandomGreedyMatchingSolver:    
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
        U = np.zeros((nrounds,numpires),dtype=np.int32)                
        V = np.zeros((nrounds,),dtype=np.int32)
        penalties = np.zeros((nrounds,),dtype=np.int32)        
        costs     = np.zeros((nrounds,),dtype=np.int32)

        
        t = 0
        while t < nrounds:
            if t == 0:
                U[0,:] = np.arange(numpires)+1 
                np.random.shuffle(U[0,:])
                t +=1
            else:                                                                  
                Rt = restraint_builder(S, U, t, d1, d2)
                Tt = travel_builder(D, S, U, t)

                uindex, gindex, violations, status = RandomicGreedySolverF(Tt, Rt, fixpenalty).solve()                        
                if status == 1:             
                    U[t, uindex] = (gindex+1)
                    V[t]         = violations                                                                
                    costs[t]     = Tt[uindex, gindex].sum()
                    penalties[t] = V[t]*fixpenalty
                    
                
                    t +=1
                else:
                    t = max(0, t-1)
        return U, V, costs, penalties, 
        
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

class RandomicGreedySolverF:
    '''
        Finds a viable solution generator for greedy matching solver    
    '''    
        
    def __init__(self, C, F, fixpenalty, nstop=10):
        
        try: 
            self.C = C
            self.F = F 
            self.fixpenalty = fixpenalty
         
            index_forbidden = F.astype('bool')
 
            C = C.astype('float')
            SC = C.sum(axis=1)
            D = (SC-C.T).T  
            D[index_forbidden] = 0 
            
            dsum = D.sum(axis=1)
            if np.isnan( dsum ).any() | (dsum == 0).any():
                self.hopeless = 1                
            else:     
                self.hopeless = 0
                P = (D.T /dsum).T
                self.P = P 
                self.nstop = nstop
                
        except Exception as e: 
            print e 
         
            
        

    def _updateP_(self, P, updaterows, removecol):
        '''
            Removes ith column updating property of P
            P[updaterows,:].sum(axis=1) = 1
        '''    
        for row in updaterows:
            pr = P[row, removecol]
            for col in xrange(P.shape[1]):
                if col == removecol:
                    P[row, col] = 0 
                elif pr <0.99:
                    P[row, col]=P[row, col]/(1-pr)     
        return P             

    def _randomic_greedy_umpireordering_(self):
        '''
            Returns an umpire ordering, most number of conflicts having the hightest probability of coming in first
        '''
        umpireconflicts  = self.F.sum(axis=1).astype('float')
        numpires         = len(umpireconflicts) 
        sumconflicts     = umpireconflicts.sum()
        umpireprob       = (umpireconflicts+1)/(sumconflicts+numpires)
        
        umpordering       = np.random.choice(np.arange(numpires),replace=False,size=numpires,p=umpireprob)
        return umpordering
             
    def solve(self):
        if self.hopeless == 1:
            return [],[], 0
        F = self.F
        fixpenalty = self.fixpenalty 
        numpires, ngames = self.P.shape
        nstop = self.nstop 
        t = 0 
        feasible = False 
        while not(feasible): 
            P = np.array(self.P) 

            indexumpires = self._randomic_greedy_umpireordering_()
            indexgames   = np.zeros((ngames,), dtype=np.int32 )
            feasible     = True 
            status       = 1
            for i,u in enumerate(indexumpires):                                    
                pu = P[u,:] 
                pu_sum = float(pu.sum())
                if ((np.isnan(pu).any()) | (pu_sum < 0.99) | (pu_sum > 1.01)):
                    t += 1
                    feasible = t > nstop 
                    status   = 0
                    break 
                else:
                    games = np.random.choice(np.arange(numpires), size=1, replace=True, p=pu)
                    g = games[0]
                  
                    indexgames[u] = g          
                    if i < numpires-1:  
                        updaterows = indexumpires[(i+1):]
                        removecol = g 
                        P = self._updateP_(P, updaterows, removecol)             

        indexumpires = np.arange(numpires)
        violations   = F[indexumpires, indexgames].sum()
        return indexumpires, indexgames, violations, status     