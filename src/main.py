'''
Created on Jun 8, 2017

@author: Varela
'''
import numpy as np
import pandas as pd  
import readers as rd  
import os
import timeit

from builders       import schedule_builder

from ga import ga_initialpopulation,ga_crossover, ga_fitness

def publish_score(nfit, individualdecile, individualbest):    
    scoredecile= "{:,}".format(int(individualdecile.score()))
    scorebest  = "{:,}".format(int(individualbest.score()))
    distancebest = "{:,}".format(int(individualbest.costs.sum()))
    print 'ga\tepoch\t%04d\ttop decile[%d]\t%s\ttop score\t%s\tBest distance\t%s' % (epochs,nfit,scoredecile,scorebest,distancebest)
    
if __name__ == '__main__':
    pass

instancename = 'umps8'
nteams, D, opponents = rd.instance_reader(instancename)
S = schedule_builder(opponents)

timestamp = int(timeit.time.time())

### GA INITIAL POPULATION
d1 = 0 
d2 = 0
epochs = 0 
replaceperc  = 0.15 
npopulation = 500
fixpenalty  = 1000
population  = ga_initialpopulation(npopulation, D, S, d1, d2, fixpenalty)

fittest     = population[0]

### GA CROSSOVER 
nfit = int(0.1 * npopulation ) 
fitalpha  = 0.5
tol = 3e-2
maxepochs = 2e1
stop_criteria = False
fitmv = tol*1000
 
fittest.persist(D, S, epochs, d1, d2, instancename, timestamp)
 
while not stop_criteria:
    population = ga_crossover(D, S, d1, d2, population, replaceperc)
       
    fitscore        = ga_fitness(population, nfit)
    stop_criteria   = (epochs > maxepochs) | (abs(fitscore - fitmv) < tol)
     
    publish_score(nfit, population[nfit-1], population[0])
    fitmv = (fitalpha)*fitscore + (1-fitalpha)*fitmv
    epochs +=1 
 
   
fittest =  population[0]
fittest.persist(D, S, epochs, d1, d2, instancename, timestamp)
  
