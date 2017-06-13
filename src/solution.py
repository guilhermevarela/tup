'''
Created on Jun 12, 2017

@author: Varela
'''
import numpy as np 
from solvers import ProbabilisticGreedyMatchingSolver

class TUPSolution(object):
    '''
    TUP solution stores d1, d2
    '''


    def __init__(self, D, S, d1, d2):                         
        cost, solution, violations =  ProbabilisticGreedyMatchingSolver(D, S, d1, d2).solve()

        self.cost = cost 
        self.solution = solution 
        self.violations = violations 
        