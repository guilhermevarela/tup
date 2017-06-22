'''
Created on Jun 9, 2017

@author: Varela
'''
from tup import TUP
import numpy as np

import copy 


def ga_initialpopulation(npopulation, D, S, d1, d2, fixcost):
    population = []
    i = 0 
    tries = 0
    for i in xrange(npopulation):
        if (i % (0.1*npopulation) == 0 ):
            print "ga_initialpopulation\tcreated\t(%03d/%03d)\ttries\t%05d" % (i,npopulation, tries)
            
        sol = TUP(D,S,d1,d2,fixcost)
        
        population.append(sol)

        
    population = ga_rank(population)    
    print "ga_initialpopulation\t(%03d/%03d)" % (npopulation,npopulation)
    return  population     

def ga_crossover(D, S, d1, d2, population, replaceperc=0.15):
    ncrossover = int(len(population)/2)
    nreplace   = int(len(population) * replaceperc)
          
    newgeneration = []
    
    
    tries   = 0
    prevbest = copy.deepcopy(population[0]) 
    while len(newgeneration) < ncrossover:                                  
        parents = np.random.choice(population,size=2,replace=False)
        solx    = copy.deepcopy(parents[0])  
        solcopy = parents[1] 
            
        solx.x(solcopy, D, S, d1, d2)

        exists =  ga_exists(newgeneration,solx) | ga_exists(population,solx) 
        if not exists:  
            newgeneration.append(solx)
        tries   +=1
        
    #replace the best 
    newgeneration   = ga_rank(newgeneration)    
    replacestart = len(population)-nreplace
    keepfinish   = nreplace
    
    fitnessbefore= ga_fitness(population, nreplace)
    
    population[replacestart:] = newgeneration[0:keepfinish]    
    population = ga_rank(population)

    if prevbest.score() < population[0].score():
        population = [prevbest] + population[:-1]
    fitnessafter=ga_fitness(population, nreplace)    

    return population



def ga_rank(population):
    '''
        Performs descending fitness order
    '''
    population.sort(key=lambda x : x.score())    
    return population

def ga_exists(population, solution):     
    duplicates = filter(lambda x : x.score() == solution.score(), population)
    result = False 
    if not(duplicates is None):         
        for d in duplicates:
            if (d.U == solution.U).all():
                result = False 
                break 
    return result            

def ga_fitness(population, nbest): 
    return np.array(map(lambda x : x.score(), population[:nbest])).mean()

