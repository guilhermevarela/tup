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

if __name__ == '__main__':
    pass

nteams, D, opponents = rd.instance_reader()
S = schedule_builder(opponents)

sol =  TUPSolution(D,S,0,0)
print sol.cost
print sol.violations
print sol.solution

# nrounds, numpires,_ = S.shape 
# U = umpires_builder(nrounds, nteams)
# U[0,:] = np.arange(numpires) +1
# np.random.shuffle(U[0,:])

# pgm = ProbabilisticGreedyMatching(D, S, 0, 0)
# pgm.solve()

# for t in xrange(1,nrounds):
#     Ct = constraint_violationmask_builder(D, S, U, t, 0, 0)
#     Tt = travel_builder(D, S, U, t)
#     print "Travel distances @ ", t 
#     print Tt
#     
#     solvers = BipartiteMatchingSolver(Tt)
#     umpires, games, c  =  solvers.solve()
#     U[t,:] = games +1
# 
#     print umpires
#     print games
#     print c 
#     print Ct
