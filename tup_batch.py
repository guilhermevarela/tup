'''
Created on Jun 23, 2017

@author: Varela
'''
import glob 
import sys 
import ga_experiment 
if __name__ == '__main__':
    pass
if len(sys.argv) < 4:
	print '%s expects at least 3 parameters instance_familyname, d1, d2 '%(sys.argv[0])
else:
	instance_familyname = sys.argv[1]
	glob_pattern = './instances/%s*.txt' % (instance_familyname)
	filenames =  glob.glob(glob_pattern)
	if filenames: 
		print 'Found files matching pattern'
		for name in filenames:
			print name 
		d1 = 	int(sys.argv[2])
		d2 = 	int(sys.argv[3])
		ga_experiment.execute(instance_familyname, filenames, int(sys.argv[2]), int(sys.argv[3])) 	
	else:
		print 'No files found matching:', instance_familyname