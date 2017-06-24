'''
Created on Jun 23, 2017

@author: Varela

Handles input output
'''
import os 
import pandas as pd 

def persist_benchmark(experimentid, timestamp, benchmark_dict, ouput_dir=''):
	'''

		Receives a benchmark dictionary and saves in ouput/benchmarks/timestamp

	'''	
	if not(ouput_dir):
		output_dir = "./output/benchmarks/%s/" %(timestamp)
 		if not os.path.exists(output_dir):
			os.makedirs(output_dir)

	df = pd.DataFrame.from_dict(benchmark_dict)     
	df = pd.DataFrame.transpose(df)
	filename = output_dir + experimentid + ".csv"
	
	df.to_csv(filename, sep=',')

def persist_solution(tupdf, D, S, epochs, q1, q2, instancename, timestamp, ouput_dir=''):
	'''

		Receives a pandas dataframe representing a tup solution

	'''	
	if not(ouput_dir):
		output_dir = "./output/%s/%d/" 
		output_dir = output_dir %(instancename,timestamp)
	
	makedir(output_dir)

	ganame      = 'ga_%s_i%04d-q1_%02d-q2_%02d.csv' % (instancename,epochs,q1,q2)
	filepath    = output_dir + ganame  
	tupdf.to_csv(filepath, sep=',')                

def persist_siteformat1(tupdf, instancename, timestamp, q1, q2, ouput_dir=''):
	'''

		Receives a pandas dataframe representing a tup solution

	'''	
	if not(ouput_dir):
		output_dir = "./output/%s/%d/" 
		output_dir = output_dir %(instancename,timestamp)

	makedir(output_dir)	
	
	exportname      = '%s_%d_%d.txt' % (instancename,q1,q2)        
	filepath        = output_dir + exportname 
	tupdf.to_csv(filepath, sep=',', header=False, index=False)                

def persist_siteformat2(tupdf, instancename, timestamp, q1, q2, ouput_dir=''):
	if not(ouput_dir):
		output_dir = "../src/output/%s/%d/" 
		output_dir = output_dir %(instancename,timestamp)

	makedir(output_dir)
	
	exportname      = '%s_%d_%d.txt' % (instancename,q1, q2)        
	filepath        = output_dir + exportname
	tupdf.to_csv(filepath, sep=' ', header=False, index=False, encoding='utf-8')          

def makedir(output_dir):
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)
	return output_dir	