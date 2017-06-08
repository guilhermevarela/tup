'''
Created on Jun 8, 2017

@author: Varela
'''
import numpy as np 
import pandas as pd
import networkx as nx 
import readers as rd  

if __name__ == '__main__':
    pass

n, dist, oppo = rd.instance_reader()
D = nx.from_numpy_matrix(dist)
print n
print dist 
print oppo
print D.number_of_nodes() 

