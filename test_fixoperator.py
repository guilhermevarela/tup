import numpy as np 
from ga import ga_initialpopulation, ga_mutation
from utils import get_populationid  
from singletons import set_dictionary



if __name__ == '__main__':

	argin = {'d1':0 ,'d2':0, 'penalty': 1000}
	dct = set_dictionary('./instances/umps8.txt',**argin)

	
	population = ga_initialpopulation(500)
	populationid = get_populationid(population)  


	scores     = np.array(map(lambda x : x.score(), population))	
	print "avg. scores before mutation", scores.sum() / 500 
	population, populationid = ga_mutation(population, populationid, 100, 300) 
	scores     = np.array(map(lambda x : x.score(), population))	
	print "avg. scores after mutation", scores.sum() / 500 
