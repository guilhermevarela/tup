'''
Created on Jun 8, 2017

@author: Varela

'''
import numpy as np
import scipy.optimize as opt

#def stable_matching():
# Python program to find maximal Bipartite matching.
# bpGraph =[[0, 1, 1, 0, 0, 0],
#         [1, 0, 0, 1, 0, 0],
#         [0, 0, 1, 0, 0, 0],
#         [0, 0, 1, 1, 0, 0],
#         [0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 1]]
#  
# g = Graph(bpGraph)
#  
# print ("Maximum number of applicants that can get job is %d " % g.maxBPM())
class PerfectMatchingSolver:

    def solve(self):
        raise NotImplementedError('Probabilistic PerfectMatchingSolver')

#Implementation of the hungarian method         
class BipartiteMatchingSolver:
    def __init__(self, C):
        self.C = C # residual graph
    
    # Ia, Ij, c = solve()
    # Ia .: index of applicants
    # Ij .: index of jobs
    # c .: total costs
    def solve(self):             
        Ia, Ij = opt.linear_sum_assignment(self.C)
        c = self.C[Ia,Ij].sum()
        return Ia, Ij, c
        