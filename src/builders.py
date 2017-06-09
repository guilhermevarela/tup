'''
Created on Jun 8, 2017

@author: Varela
'''
import numpy as np 

def bipartite_builder(nteams, D, Ut, Ht, Vt):
# Builds a bipartite graph connecting Umpires (Ut) to Gt (Ht,Vt)
# Ht home venue, Vt visitant

    if not Ut:
        Ut = np.random.permutation(int(nteams/2))
    
    u = Ut -1
    s = St -1 
                 
    return 
