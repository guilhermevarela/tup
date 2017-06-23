'''
Created on Jun 8, 2017

@author: Varela
'''
import numpy as np
import pandas as pd  
import readers as rd  
import os

from utils import *
from ga import *

def publish_score(nfit, individualdecile, individualbest):    
    scoredecile = "{:,}".format(int(individualdecile.score()))
    scorebest = "{:,}".format(int(individualbest.score()))
    distancebest = "{:,}".format(int(individualbest.travel()))
    msg = 'ga\tepoch\t%04d\ttop decile[%d]\t%s\t'
    msg +='top score\t%s\tBest distance\t%s'
    buff = msg % (epochs, nfit, scoredecile, scorebest, distancebest)
    print buff

if __name__ == '__main__':
    pass

instancename = 'umps14'
# instancename = 'umps10C'
# instancename = 'umps4'
# instancename = 'umps12'
# instancename = 'umps6'
# instancename = 'umps6A'
# instancename = 'umps8A'

nteams, D, opponents = rd.instance_reader(instancename)

S = get_schedule(opponents)
numps = int(nteams/2)


### GA INITIAL POPULATION
d1 = 0
d2 = 0
q1 = numps - d1 
q2 = int(numps/2) - d2 
epochs = 0
replaceperc = 0.15 
mutateperc  = 0.05
npopulation = 500
fixpenalty = 1000 * numps * numps   
population = ga_initialpopulation(npopulation, D, S, q1, q2, fixpenalty)
parentid = get_populationid(population)
timestamp = get_timestamp()
fittest = population[0]


#GA CROSSOVER 
nfit = int(0.1 * npopulation ) 
fitalpha  = 0.5
tol = 3e-2
maxepochs = 2e2
stop_criteria = False
fitmv = tol*1000
nreplace = int(npopulation * replaceperc)
nmutate = int(npopulation * mutateperc)

fittest.persist(D, S, epochs, q1, q2, instancename, timestamp)
publish_score(nfit, population[nfit-1], fittest)

while not stop_criteria:
    # preserve ids of the individuals for mutations
    ga_fittest_store(population)
    population = ga_crossover(D, S, q1, q2, population, replaceperc)

    population, parentid = ga_mutation(D, S, q1, q2, population, parentid, nreplace, nmutate )

    population, parentid =ga_fittest_recall(population, parentid)

    epochs +=1 
    fitscore        = ga_fitness(population, nfit)
    stop_criteria   = (epochs > maxepochs) | (abs(fitscore - fitmv) < tol)

    publish_score(nfit, population[nfit-1], population[0])
    fitmv = (fitalpha)*fitscore + (1-fitalpha)*fitmv
    

fittest =  population[0]
publish_score(nfit, population[nfit-1], fittest)
fittest.persist(D, S, epochs+1, q1, q2, instancename, timestamp)
fittest.export2(S, instancename, timestamp, q1, q2)
