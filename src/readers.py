'''
Created on Jun 8, 2017

@author: Varela
'''
import numpy as np 
import pandas as pd  
import re 

def instance_reader(instancename='umps8'):
    path  = './src/instances/'
    filepath = path + instancename + ".txt"
    
    matcher_nteams       = lambda x : re.match(r'nTeams=(.[^;]*)', x )
    searcher_distance    = lambda x : re.search(r'dist=', x)
    searcher_opponents   = lambda x : re.search(r'opponents=', x)
    matcher_array        = lambda x : re.search(r'\[(.*)\]', x)
    
    searchteams = True
    searchdist  = False  
    searchopp   = False  
    distmtrx=[]
    oppmtrx=[]
    with open(filepath, 'r') as f: 
        for line in f:
            if searchdist:
                matched = matcher_array(line)  
                if matched:
                    strarray = matched.group(1).split(' ')
                    strarray = [x for x in strarray if x] 
                    distmtrx.append( [int(x) for x in strarray])                     
                else:
                    searchdist = False  
            if searchopp:
                matched = matcher_array(line)  
                if matched:
                    strarray = matched.group(1).split(' ')
                    strarray = [x for x in strarray if x]                     
                    oppmtrx.append( [int(x) for x in strarray]) 
                else:
                    searchopp = False  
                                  
            if searchteams:                    
                n = matcher_nteams(line)
                if n: 
                    n = int(n.group(1))
                    searchteams = False                            
            else: 
                dist = searcher_distance(line)
                if dist:
                    searchdist = True
                else: 
                    opp = searcher_opponents(line)
                    if opp: 
                        searchopp = True 
                        

    distmtrx = np.array(distmtrx)          
    oppmtrx  = np.array(oppmtrx)       

    return n,distmtrx,oppmtrx
            