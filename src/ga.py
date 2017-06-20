'''
Created on Jun 9, 2017

@author: Varela
'''
from tup import TUP
import numpy as np
import signal
ga_abort = False
def ga_abort_individual(signum, frame):
        print 'aborting individual ', signum
        global ga_abort
        ga_abort = True 

def ga_initialpopulation(npopulation, D, S, d1, d2, verbose=True):
    population = []
    i = 0 
    tries = 0
    while (i < npopulation):
        tries +=1
        if verbose & (tries % (0.1*npopulation) == 0 ):
            print "ga_initialpopulation\tcreated\t(%03d/%03d)\ttries\t%05d" % (i,npopulation, tries)
            
        signal.signal(signal.SIGALRM, ga_abort_individual)
        signal.alarm(5)                             
        sol = TUP(D,S,d1,d2)
        
        global ga_abort 
        if not(ga_abort):
            population.append(sol)
            i+=1 
        else: 
            ga_abort = False    
        signal.alarm(0)
        
    population = ga_rank(population)
    if verbose: 
        print "ga_initialpopulation\t(%03d/%03d)" % (npopulation,npopulation)
    return  population     

def ga_crossover(D, S, d1, d2, population, replaceperc=0.15):
    ncrossover = int(len(population)/2)
    nreplace   = int(len(population) * replaceperc)
          
    newgeneration = []
    
    while len(newgeneration) < ncrossover:
        parents = np.random.choice(population,size=2,replace=False)
        solx    = parents[0]
        solcopy = parents[1] 
            
        solx.x(solcopy, D, S, d1, d2)

        exists =  ga_exists(newgeneration,solx) | ga_exists(population,solx) 
        if not exists:
            newgeneration.append(solx)
    
    #replace the best 
    newgeneration   = ga_rank(newgeneration)    
    replacestart = len(population)-nreplace
    keepfinish   = nreplace
    population[replacestart:] = newgeneration[0:keepfinish]
    population = ga_rank(population)

    return population

def ga_rank(population):
    '''
        Performs descending fitness order
    '''
    population.sort(key=lambda x : x.score)    
    return population

def ga_exists(population, solution):     
    duplicates = filter(lambda x : x.score == solution.score, population)
    result = False 
    if not(duplicates is None):         
        for d in duplicates:
            if (d.U == solution.U).all():
                result = False 
                break 
    return result            

def ga_fitness(population, nbest): 
    return np.array(map(lambda x : x.score, population[:nbest])).mean()