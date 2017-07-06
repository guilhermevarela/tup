'''
Created on Jun 9, 2017

@author: Varela
'''
from tup import TUP
from singletons import get_dictionary 
import numpy as np

import copy

gl_fittest = None 

#def ga_initialpopulation(npopulation, D, S, q1, q2, fixpenalty, order=False):
def ga_initialpopulation(npopulation, order=False):
  '''

    Retuns a list[npopulation] of tuple instances
    
  ''' 
  # Initialization 
  dct  = get_dictionary() 
  D    = dct['D']
  S    = dct['S']
  q1   = dct['q1']
  q2   = dct['q2']
  fixpenalty = dct['fixpenalty']
  homegames = dct['homegames']

  population = []
  i = 0 
  
  msg = "ga_initialpopulation\tcreated\t(%03d/%03d)\t"
  for i in xrange(npopulation):
      if (i % (0.1*npopulation) == 0 ):
          buff = msg %  (i,npopulation)
          print buff
          
      sol = TUP(D,S,q1,q2,fixpenalty,homegames)      
      
      population.append(sol)
      
  
  population = ga_rank(population)    
  print msg % (npopulation,npopulation)
  return  population     

#def ga_crossover(D, S, q1, q2, population, replaceperc=0.15):
def ga_crossover(population, replaceperc=0.15):
  dct = get_dictionary()

  D   = dct['D']
  S   = dct['S']
  q1  = dct['q1']
  q2  = dct['q2']
  homegames = dct['homegames']

  ncrossover = int(len(population)/2)
  nreplace   = int(len(population) * replaceperc)
        
  newgeneration = []
  

  while len(newgeneration) < ncrossover:                                  
      parents = np.random.choice(population,size=2,replace=False)
      solx    = copy.deepcopy(parents[0])  
      solcopy = parents[1] 
          
      solx.x(solcopy, D, S, q1, q2, homegames)

      exists =  ga_exists(newgeneration,solx) | ga_exists(population,solx) 
      if not exists:  
          newgeneration.append(solx)

      
  # Replace the inferirior solutions in 
  newgeneration = ga_rank(newgeneration)    
  replacestart = len(population)-nreplace
  keepfinish   = nreplace
  
  population[replacestart:] = newgeneration[0:keepfinish]    
  population = ga_rank(population)

  return population

#def ga_mutation(D, S, q1, q2, population, parentid, nreplace, nmutation):
def ga_mutation(population, parentid, nreplace, nmutation):
  dct = get_dictionary()

  D   = dct['D']
  S   = dct['S']
  q1  = dct['q1']
  q2  = dct['q2']
  homegames= dct['homegames']
  # We get only the parents who are garanteed not to have been replaced
  nparent    = int(len(population) - nreplace)  
  mutationid = np.random.choice(parentid[:nparent],size=nmutation,replace=False)
  populationid = np.array(map(id,population))
  indexes = np.in1d(populationid, mutationid,assume_unique=True) # returns mask referent to booleans
  indexes = np.where(indexes)[0]
  
  for i in indexes:
    #if population[i].P.any():          
      #population[i].repair(D, S, q1, q2, verbose=False)    
    #  population[i].sa(D, S, q1, q2, 100)   
    # else: 
    population[i].mutate(D, S, q1, q2, homegames)    


  population, populationid = ga_rank(population, populationid)
  
  return population, populationid


def ga_rank(population, populationid=None):
  '''
    Performs descending fitness order
  '''
  
  if populationid is None: 
    population.sort(key=lambda x : x.score())
    return population
  else:     
    # sorts populationid based on population keys
    populationid = [y for (x,y) in sorted(zip(population, populationid), key=lambda t : t[0].score())]
    population.sort(key=lambda x : x.score())    

    return population, populationid
  

def ga_exists(population, solution):     
  duplicates = filter(lambda x : x.score() == solution.score(), population)
  result = False 
  if not(duplicates is None):         
      for d in duplicates:
          if (d.U == solution.U).all():
              result = False 
              break 
  return result            

def ga_fitness(population, nbest): 
  return np.array(map(lambda x : x.score(), population[:nbest])).mean()


def ga_store(population):
  '''
    Stores a deepcopy  of the fittest individual
  '''
  global gl_fittest
  gl_fittest = copy.deepcopy(population[0])  
  return gl_fittest

def ga_recall(population, populationid): 
  '''
    Performs replacement only if the new fitness function is less then the parent's 
  '''
  global gl_fittest
  if gl_fittest.score() < population[0].score():
    population = [gl_fittest] + population[:-1]
    populationid = [id(gl_fittest)] + populationid[:-1]
  return population, populationid  
