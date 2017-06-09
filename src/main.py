'''
Created on Jun 8, 2017

@author: Varela
'''
import numpy as np 
import readers as rd  
 
from solver         import BipartiteMatchingSolver
from builders       import travel_builder, schedule_builder, umpires_builder


if __name__ == '__main__':
    pass

nteams, distances, opponents = rd.instance_reader()


S = schedule_builder(opponents)
nrounds = S.shape[0]
U = umpires_builder(nrounds, nteams)
U[0,:] = np.arange(int(nteams/2)) +1
np.random.shuffle(U[0,:])
t = 1

Tt = travel_builder(distances, S, U, t, nteams)
print "Travel distance"
print Tt

solver = BipartiteMatchingSolver(Tt)
umpires, games, c  =  solver.solve()
  

Uy = S[0,:,:]


print umpires
print games
print c 
 
