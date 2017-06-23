
import numpy as np
import pandas as pd  
import readers as rd  
import os
from utils import get_populationid, get_schedule, get_timestamp

from ga import * 


def execute(instance_family, d1, d2):
  epochs = 0
  replaceperc = 0.15 
  mutateperc  = 0.05
  npopulation = 500
  

  nreplace = int(npopulation * replaceperc)
  nmutate = int(npopulation * mutateperc)
	# ndecile = int(0.1 * npopulation )
  fitalpha  = 0.5

  tol = 5e-3
  maxepochs = 2e2
  stop_criteria = False
  fitmv = tol*1000

  for instancepath in instance_family: 
    instancename = instancepath.split('/')[-1]     
    instancename = instancename.split('.')[0]     
    nteams, D, opponents = rd.instance_reader(instancepath)
    numps = int(nteams/2)
    fixpenalty = 1000 * numps * numps   
    S = get_schedule(opponents)

    q1 = numps - d1 
    q2 = int(numps/2) - d2 

    population = ga_initialpopulation(npopulation, D, S, q1, q2, fixpenalty)
    parentid = get_populationid(population)
    timestamp = get_timestamp()
    # fittest = population[0]
    stopexperiment = False 
    record(locals(),export=False)
    while not stopexperiment:
      # preserve ids of the individuals for mutations
      ga_fittest_store(population)
      population = ga_crossover(D, S, q1, q2, population, replaceperc)

      population, parentid = ga_mutation(D, S, q1, q2, population, parentid, nreplace, nmutate )

      population, parentid = ga_fittest_recall(population, parentid)

    
      
      # fitdecile        = population[ndecile]
      # fittest          = population[0]


      stopexperiment = eval_stopcriteria(instancename, population, tol, epochs, maxepochs)

  
      # fitmv = (fitalpha)*fitscore + (1-fitalpha)*fitmv
      epochs +=1 
    

	# fittest =  population[0]
	# publish_score(ndecile, population[ndecile-1], fittest)
	# fittest.persist(D, S, epochs+1, q1, q2, instancename, timestamp)
	# fittest.export2(S, instancename, timestamp, q1, q2)
  record(locals(), export=True) 
	
	
def eval_stopcriteria(instancename, population, tol, epochs, maxepochs):		
  result = False 

  npopulation = len(population) 
  fitdecile = population[int(0.1*npopulation)]
  fittest   = population[0]

  publish(instancename, epochs, fitdecile, fittest)
  diversity = float(fitdecile.score() - fittest.score()) / fitdecile.score() 
  
  if diversity <= tol: 
    result = True 
    print 'ga_%s stop achieved due low diversity' % (instancename)
  elif  epochs >= maxepochs:   
    result = True 
    print 'ga_%s_stop achieved due to max epochs' % (instancename)
  return result    



def record(vars, export=False):
  D = vars['D']
  S = vars['S']
  epochs =vars['epochs']
  q1 =vars['q1']
  q2 =vars['q2']
  instancename =vars['instancename']
  timestamp =vars['timestamp']
  population = vars['population']
  
  population[0].persist(D, S, epochs, q1, q2, instancename, timestamp)
  if export:
    population[0].export2(S, instancename, timestamp, q1, q2)    

def publish(instancename, epochs, individualdecile, individualbest):    
    scoredecile = "{:,}".format(int(individualdecile.score()))
    scorebest = "{:,}".format(int(individualbest.score()))
    distancebest = "{:,}".format(int(individualbest.travel()))

    msg = 'ga_%s\tepoch\t%04d\ttop 10%%\t%s\t'
    msg +='top score\t%s\tBest distance\t%s'
    buff = msg % (instancename, epochs, scoredecile, scorebest, distancebest)
    print buff

