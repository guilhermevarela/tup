import numpy as np 
import timeit

def get_populationid(population):
	return np.array(map(id, population))

def get_timestamp():
	return int(timeit.time.time())

def get_schedule(opponents):
    # Converts the 2D opponents matrix from files to 3D
    nrounds, nteams = opponents.shape
    newshape = (nrounds,int(nteams/2))
    
    #home index are those index that have a positive value
    I = np.tile(np.arange(nteams).reshape(1,nteams),(nrounds,1))
    homerows, homecols  = np.where(opponents > 0)
    H = I[homerows,homecols]+1         # from index to teams
    H = H.reshape(newshape)            #official schedule 
      
    #visiting index are those index that have a negative value
    maskv  = opponents > 0    # from index to teams
    V      = opponents[maskv]     # linearized
    V      = V.reshape(newshape)
    
    S     = np.zeros((nrounds,int(nteams/2),2), dtype=np.int32)
    S[:,:,0]     = H 
    S[:,:,1]     = V 
    return S	