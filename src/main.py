'''
Created on Jun 8, 2017

@author: Varela
'''
import numpy as np 
import pandas as pd
import networkx as nx 
import readers as rd  
from solver import BipartiteMatchingSolver
from datatranform import opponents_transform
if __name__ == '__main__':
    pass

n, dist, oppo = rd.instance_reader()

print n
print dist 
print oppo

# D =[[0, 1, 1, 0, 0, 0],
#     [1, 0, 0, 1, 0, 0],
#     [0, 0, 1, 0, 0, 0],
#     [0, 0, 1, 1, 0, 0],
#     [0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 1]]
#    
# s = BipartiteMatchingSolver(D)
# print ("Maximum number of applicants that can get job is %d " % s.maxBPM()) 
#  
S = opponents_transform(oppo)
print S
 

