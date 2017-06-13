'''
Created on Jun 8, 2017

@author: Varela
'''
import numpy as np 
import readers as rd  
 
# from solvers         import * 
# from builders       import schedule_builder, umpire_builder
from builders       import schedule_builder
from solution import TUPSolution
from ga import *
if __name__ == '__main__':
    pass

nteams, D, opponents = rd.instance_reader()
S = schedule_builder(opponents)

d1 = 0 
d2 = 0 
replaceperc=0.15
npopulation = 500
population = ga_initialpopulation(npopulation, D, S, d1, d2)
t = 0
nfit = 30
fitalpha  = 0.5
tol = 3e-2
tmax = 5e2
stop_criteria = False
fitmv = tol*1000 
while not stop_criteria:
    population = ga_crossover(D, S, d1, d2, population, replaceperc)

    fitscore = ga_fitness(population, nfit)
    stop_criteria = t > tmax & (abs(fitscore - fitmv) < tol) 
    fitmv = (fitalpha)*fitscore + fitalpha*fitmv 