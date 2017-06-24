
import readers as rd  
from utils import get_populationid, get_schedule, get_timestamp
from writers import persist_benchmark
from ga import * 

#Dictionary that stores experiment settings
gl_benchmark = {}

def execute(familyname,filepaths, d1, d2):
  epochs = 0
  replaceperc = 0.15 
  mutateperc  = 0.05
  npopulation = 500
  

  nreplace = int(npopulation * replaceperc)
  nmutate = int(npopulation * mutateperc)
  fitalpha  = 0.5

  tol = 5e-3
  maxepochs = 2e2
  stop_criteria = False
  
  timestamp = get_timestamp()
  experimentid = familyname + '_' + str(timestamp)
  for instancepath in filepaths: 
    instancename = instancepath.split('/')[-1]     
    instancename = instancename.split('.')[0]     
    nteams, D, opponents = rd.instance_reader(instancepath)
    numps = int(nteams/2)
    fixpenalty = 1000 * numps
    S = get_schedule(opponents)

    q1 = numps - d1 
    q2 = int(numps/2) - d2 

    instancetimestamp = get_timestamp()
    population = ga_initialpopulation(npopulation, D, S, q1, q2, fixpenalty)
    parentid = get_populationid(population)

    stopexperiment = False 
    epochs = 0
    record_instance(locals(),export=False)
    while not stopexperiment:
      # preserve ids of the individuals for mutations
      ga_store(population)
      population = ga_crossover(D, S, q1, q2, population, replaceperc)

      population, parentid = ga_mutation(D, S, q1, q2, population, parentid, nreplace, nmutate )

      population, parentid = ga_recall(population, parentid)
      
      benchmark(locals())
      stopexperiment = eval_stopcriteria(instancename, population, tol, epochs, maxepochs)

      epochs +=1 
      

    record_instance(locals(), export=True) 
  record_benchmark(experimentid, timestamp)
	
	
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



def record_instance(localvars,export=False):
  D = localvars['D']
  S = localvars['S']
  epochs =localvars['epochs']
  q1 =localvars['q1']
  q2 =localvars['q2']
  instancename =localvars['instancename']
  timestamp =localvars['timestamp']
  population = localvars['population']
    
  
  population[0].persist(D, S, epochs, q1, q2, instancename, timestamp)
  if export:
    population[0].export2(S, instancename, timestamp, q1, q2)    

def record_benchmark(experimentid, timestamp):
    global gl_benchmark 
    persist_benchmark(experimentid, timestamp, gl_benchmark)

def benchmark(localvars):
  fittest = localvars['population'][0]
  instancetimestamp = localvars['instancetimestamp']
  global gl_benchmark

  update = False 
  if gl_benchmark.has_key(instancetimestamp):
    #update only if best has improved
    prevscore = gl_benchmark[instancetimestamp]['score']
    if prevscore >  fittest.score():
      update = True 
  else:
      update = True 

  if update:  
    thistimestamp = get_timestamp()
    instancename = localvars['instancename']
    gl_benchmark[instancetimestamp] = dict(
      [
        ('timestamp',thistimestamp), ('delta',thistimestamp-instancetimestamp),
          ('violations',fittest.violations()), ('score',fittest.score()), ('travel',fittest.travel()),
            ('instancename', instancename)
      ]
    )

def publish(instancename, epochs, individualdecile, individualbest):    
    scoredecile = "{:,}".format(int(individualdecile.score()))
    scorebest = "{:,}".format(int(individualbest.score()))
    distancebest = "{:,}".format(int(individualbest.travel()))

    msg = 'ga_%s\tepoch\t%04d\ttop 10%%\t%s\t'
    msg +='top score\t%s\tBest distance\t%s'
    buff = msg % (instancename, epochs, scoredecile, scorebest, distancebest)
    print buff

