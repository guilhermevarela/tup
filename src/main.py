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

def publish_score(scoremean, scorebest):    
    strfit_mean= "{:,}".format(int(scoremean))
    strfit_best= "{:,}".format(int(scorebest))
    print 'ga\tepoch\t%04d\ttop score mean[%d]\t%s\ttop score\t%s' % (epochs,nfit,strfit_mean,strfit_best)
    
if __name__ == '__main__':
    pass

instancename = 'umps8'
nteams, D, opponents = rd.instance_reader(instancename)
S = schedule_builder(opponents)

timestamp = int(timeit.time.time())

### GA INITIAL POPULATION
d1 = 1 
d2 = 1
epochs = 0 
replaceperc  = 0.15
npopulation = 500
fixpenalty  = 100
population  = ga_initialpopulation(npopulation, D, S, d1, d2, fixpenalty)
fittest     = population[0]

### GA CROSSOVER 
nfit = 30
fitalpha  = 0.5
tol = 3e-2
maxepochs = 5e1
stop_criteria = False
fitmv = tol*1000

fittest.persist(D, S, epochs, d1, d2, instancename, timestamp)

while not stop_criteria:
    population = ga_crossover(D, S, d1, d2, population, replaceperc)
      
    fitscore        = ga_fitness(population, nfit)
    stop_criteria   = (epochs > maxepochs) | (abs(fitscore - fitmv) < tol)
    
    publish_score(fitscore, population[0].score)
    fitmv = (fitalpha)*fitscore + (1-fitalpha)*fitmv
    epochs +=1 

  
fittest =  population[0]
fittest.persist(D, S, epochs, d1, d2, instancename, timestamp)
 
