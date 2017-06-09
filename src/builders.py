'''
Created on Jun 8, 2017

@author: Varela
'''
import numpy as np 

def travel_builder(D, S, U, t, nteams):
    # Builds a matrix for maximum flow algorithm     
#     origin      = U[:,0,-1]-1 # From last UMPIRE LOCATIONS
    nmatches = int(nteams/2)
    origin      = U[:,0]      # From last UMPIRE LOCATIONS
    destination = S[t,:, 0]   # From next GAME LOCATIONS
    # from teams to index mappping 
    origin -=1
    destination -=1
    #index with all combinations
    R = np.tile(origin.reshape(nmatches,1),(1,nmatches))
    C = np.tile(destination.reshape(1,nmatches),(nmatches,1)) 
         
    Tt = D[R, C].T   
    return Tt  

             
