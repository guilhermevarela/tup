

#import numpy as np 
import copy as cp 
from readers import instance_reader
from ga import ga_initialpopulation, ga_mutation, ga_rank  
from utils import get_populationid, get_schedule 

if __name__ == '__main__':

	nteams, D, S = instance_reader('umps8.txt','./instances/')
	S = get_schedule(S)


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

	psum=0
	for p in parent:
		psum +=p.score() 		

	print 'before sa averange population score'	, float(psum)/npopulation

	population, populationid = ga_mutation(D, S, q1, q2, parent , parentid, nreplace, nmutation)
	
	psum=0
	for p in population:
		psum +=p.score() 			
	print 'after sa averange population score'	, float(psum)/npopulation	
	



