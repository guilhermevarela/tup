'''
Created on Jul 5, 2017

@author: Varela

Defined by instances and running parameters

'''

from readers import instance_reader
from utils import get_schedule 
gl_dict = {}


def get_dictionary():
	'''

		gets data dictionary

	'''	
	global gl_dict 
	return gl_dict

def set_dictionary(instancename, **kwargs):	
	'''

		sets the dictionary to hold the instancedata it adds extra parameters based on kwargs
		d1 -> q1, d2 -> q2, penalty -> fixpenalty

	'''	
	dctnry = get_dictionary()
	nteams, D, O, = instance_reader(instancename)
	numps = int(nteams/2)
	ngames = numps
	S = get_schedule(O)
	nrounds = S.shape[0]
	dctnry.update({'D': D, 'S': S, 'nteams': nteams, 'numps': numps, 'ngames': ngames, 'nrounds': nrounds})
	if kwargs:
		dctnry.update(kwargs)
		
		if kwargs.has_key('d1'):
			dctnry['q1'] = numps - kwargs['d1']
		
		if kwargs.has_key('d2'):
			dctnry['q2'] = int(numps/2) - kwargs['d2']
		
		if kwargs.has_key('penalty'):			
			dctnry['fixpenalty'] = numps*kwargs['penalty']

	return dctnry

def dictionary_publish(**kwargs):
	dctnry = get_dictionary() 
	dctnry.update(kwargs)
	return dctnry	


