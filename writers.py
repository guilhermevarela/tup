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
	print df
	filename = output_dir + experimentid + ".csv"
	
	df.to_csv(filename, sep=',')