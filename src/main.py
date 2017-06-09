'''
Created on Jun 8, 2017

@author: Varela
'''
import numpy as np 
import readers as rd  
 
from solver         import BipartiteMatchingSolver
from datatransform  import opponents_transform
from builders       import travel_builder


if __name__ == '__main__':
    pass

n, dist, oppo = rd.instance_reader()

# print n
# print dist 
# print oppo

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
Uy = S[0,:,:]
Tt = travel_builder(dist, S, Uy, 1, n)
print "Travel distance"
print Tt

# uindex, gindex = opt.linear_sum_assignment(Tt)

# print Tt[uindex,gindex].sum() 
solver = BipartiteMatchingSolver(Tt)
umpires, games, c  =  solver.solve()

print umpires
print games
print c 
# print ("Maximum number of applicants that can get job is %d " % s.solve())
 
