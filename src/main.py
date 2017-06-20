'''
Created on Jun 8, 2017

@author: Varela
'''
import numpy as np
import pandas as pd  
import readers as rd  
import os
import timeit
# from solvers         import * 
# from builders       import schedule_builder, umpire_builder
from builders       import schedule_builder
from tup import TUP
from ga import *
if __name__ == '__main__':
    pass


nteams, D, opponents = rd.instance_reader()
S = schedule_builder(opponents)

timestamp = int(timeit.time.time())

output_dir = "../output/%d/" %(timestamp)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
  
d1 = 0 
d2 = 0 
replaceperc  = 0.15
npopulation = 100
population  = ga_initialpopulation(npopulation, D, S, d1, d2)
df          = population[0].to_frame(D,S)
filepath    = output_dir + 'ga_population0000-8.csv'  
df.to_csv(filepath, sep=',')

# t = 0
# nfit = 30
# fitalpha  = 0.5
# tol = 3e-2
# tmax = 5e2
# stop_criteria = False
# fitmv = tol*1000 
# while not stop_criteria:
#     population = ga_crossover(D, S, d1, d2, population, replaceperc)
#      
#     fitscore   = ga_fitness(population, nfit)
#     stop_criteria = (t > tmax) | (abs(fitscore - fitmv) < tol) 
#     fitmv = (fitalpha)*fitscore + (1-fitalpha)*fitmv
#     t +=1 
#  
# df = population[0].to_frame(D,S)
#  
# fname =  'ga_population%04d-8.csv' % (t)
# filepath    = output_dir + fname  
# df.to_csv(filepath, sep=',')     