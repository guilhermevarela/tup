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

instancename = 'umps8'
nteams, D, opponents = rd.instance_reader(instancename)
S = schedule_builder(opponents)

timestamp = int(timeit.time.time())

output_dir = "../src/output/%s/%d/" %(instancename,timestamp)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
  
d1 = 2 
d2 = 1 
replaceperc  = 0.15
npopulation = 500
population  = ga_initialpopulation(npopulation, D, S, d1, d2)
df          = population[0].to_frame(D,S)
ganame      = 'ga_%s_i%04d-d1_%02d-d2_%02d.csv' % (instancename,0,d1,d2)
filepath    = output_dir + ganame  
df.to_csv(filepath, sep=',')

t = 0
nfit = 30
fitalpha  = 0.5
tol = 3e-2
tmax = 5e2
stop_criteria = False
fitmv = tol*1000 
while not stop_criteria:
    population = ga_crossover(D, S, d1, d2, population, replaceperc)
      
    fitscore   = ga_fitness(population, nfit)
    stop_criteria = (t > tmax) | (abs(fitscore - fitmv) < tol)
    fitscore_str    = "{:,}".format(int(fitscore))
    bestscore_str   = "{:,}".format(int(population[0].score))
    print 'ga\tepoch\t%04d\ttop score mean[%d]\t%s\ttop score\t%s' % (t,nfit,fitscore_str,bestscore_str)
     
    fitmv = (fitalpha)*fitscore + (1-fitalpha)*fitmv
    t +=1 
  
df = population[0].to_frame(D,S)
  
ganame      = 'ga_%s_i%04d-d1_%02d-d2_%02d.csv' % (instancename,t,d1,d2)
filepath    = output_dir + ganame      
df.to_csv(filepath, sep=',')     