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
from ga import ga_initialpopulation
if __name__ == '__main__':
    pass

nteams, D, opponents = rd.instance_reader()
S = schedule_builder(opponents)

# sol =  TUPSolution(D,S,0,0)
# print sol.cost
# print sol.violations
# print sol.solution

npopulation = 500
population = ga_initialpopulation(npopulation, D, S, 0, 0)
temppopulation = []
t = 0
keep_searching  = True  
while keep_searching:
    
     

