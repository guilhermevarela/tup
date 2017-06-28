import numpy as np 
import copy as cp 
from readers import instance_reader
from ga import ga_initialpopulation, ga_mutation, ga_rank  
from utils import get_populationid, get_schedule 
import umps as up
from scipy.sparse import csr_matrix 
import networkx	as nx 

def pointwise_repair(D, S, q1, q2, tup, verbose=True):
	'''
	Applies hopcroft_karp_matching algorithm
	'''
	# Stores the indexes of the forbidden teams
	U  = tup.U 		
	V4  = tup.V4 
	V5  = tup.V5 
	nteams = D.shape[0]
	numps  = int(nteams/2)

	idt = np.arange(S.shape[0])
	idt = idt[(V4+V5).sum(axis=1)>0]
	t = np.random.choice(idt,size=1)[0]  # has to be randomly drawn
	

	idu = np.arange(0,numps).reshape(1,numps)
	#FORBBIDEN BY VIOLATION 4
	F4 = np.zeros((nteams, numps))	
	Hom = up.umps2home(S,U)
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

	Adv = up.umps2adversaries(S,U)
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
	S = csr_matrix(A)
	# Game vertices are shifted by numps
	G = nx.bipartite.from_biadjacency_matrix(S)
	dct = nx.bipartite.hopcroft_karp_matching(G)

	if verbose: 
		nviolprev= (V4[t,:] + V5[t,:]).any().sum()
		nviolnew = numps - int(len(dct.keys())/2)
		print "number of violations %d -> %d" %(nviolprev, nviolnew)
		

	# Defines new allocation - filling the positions 
	# in case no viable answer was found 
	idg = np.zeros((numps,))
	masku = np.ones((numps,), dtype=bool)
	maskg = np.ones((numps,), dtype=bool)

	#import code; code.interact(local=dict(globals(), **locals()))
	for u, g in dct.items():
		if u < numps:
			idg[u] = int(g - numps) # We have to adjust after transformation
			masku[u] = False 
			maskg[g-numps]= False
	
	# fills left overs in a sequential way		
	idg[maskg] = U[t,masku]		
	return idu, idg 		


	
if __name__ == '__main__':

	nteams, D, S = instance_reader('umps8.txt','./instances/')
	S = get_schedule(S)

	import code; code.interact(local=dict(globals(), **locals()))

	nrounds  = S.shape[0] 
	numps  = int(nteams/2)

	d1 = 0 
	d2 = 0 
	q1 = numps - d1 
	q2 = int(numps/2)-d2 
	
	fixpenalty = 1000* numps 
	npopulation = 500
	nmutation  = int(0.05 * npopulation) 
	nreplace   = int(0.15  * npopulation)
	population = ga_initialpopulation(500, D, S, q1, q2, fixpenalty)
	parentid = get_populationid(population)
	parent, parentid = ga_rank(population,parentid)

	pointwise_repair(D, S, q1, q2,  cp.deepcopy(population[0]))