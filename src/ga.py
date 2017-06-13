'''
Created on Jun 9, 2017

@author: Varela
'''
from solution import TUPSolution
import numpy as np
def ga_initialpopulation(npopulation, D, S, d1, d2):
    population = []
    for _ in xrange(npopulation):
        sol = TUPSolution(D,S,d1,d2)
        population.append(sol) 
    return  population     

def ga_crossover(population, replaceperc=0.15, D, S, d1, d2):
    ncrossover = int(population/2)
    
    nrounds, numpires, _  = S.shape  
    temp_population = []
    
    parents = np.random.choice(population,size=2,replace=False)
    p1 = parents[0]
    p2 = parents[1] 
    
    solx      = population[p1]
    solc      = population[p2]
    solx.x(solc, D, S, d1, d2):