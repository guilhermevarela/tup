import numpy as np 

def umps2home(S, U):
	'''

		umps2homevenues hometeams / homevenues given an umps assignment

	'''	
	#Home venues within schedule
	nrounds, nseries = U.shape 
	H = S[:,:,0]

	rows = np.tile(np.arange(nrounds).reshape((nrounds,1)),(1,nseries))
	cols = U-1
	return H[rows,cols]

def umps2adversaries(S, U):
	'''

		umps2adversaries returns the adversary team an umps assignent

	'''	
	#adversary team within schedule
	nrounds, nseries = U.shape 
	A = S[:,:,1]

	rows = np.tile(np.arange(nrounds).reshape((nrounds,1)),(1,nseries))
	cols = U-1
	return A[rows,cols]

def umps2violations3(S,U):
	'''
		umps2violations3 every umpire sees a team at least once at home

	'''	
	H = umps2home(S, U)		
	teams = np.amax(H)

	V3 = np.zeros(H.shape, dtype=np.int32) 
	for h in xrange(H.shape[1]):
		teamsvisited = np.unique(H[:,h])
		V3[-1,h] = teams - len(teamsvisited)
	
	return V3 		

def umps2violations4(S, U, q1):
	'''

		umps2violations4 no umpire is in home site more then once in q1=numps-d1 periods

	'''	
	# Performs q1-1 shifts in order to compute V4
	V4 = np.zeros(U.shape, dtype=np.int32)
	H = umps2home(S,U)	
	Q = H 
	for q in xrange(1,q1):
		Q = np.roll(Q, 1, axis=0) 
		Q[:q,:]=-1
		V4 += (H == Q).astype('int')
	return V4 	


def umps2violations5(S, U, q2):
	'''

		umps2violations5 no umpire sees a team more then once in q2=int(numps/2)-d2 consecutive slots

	'''	
	# Performs q2-1 shifts in order to compute V5
	V5 = np.zeros(U.shape, dtype=np.int32)
	H = umps2home(S,U)	
	A = umps2adversaries(S,U)	
	QH = H
	QA = A 
	for q in xrange(1,q2):
		#Compare with previous period
		QH = np.roll(QH, 1, axis=0) 
		QA = np.roll(QA, 1, axis=0) 
		#Ignores first line - which was rotated
		QH[0,:]=-1
		QA[0,:]=-1
		V5 += (H == QH).astype('int') + (A == QA).astype('int') 

	return V5

def umps2travel(D, S, U):
	'''

		umps2travel computes the traveling distance for each empire for the period

	'''	
	nrounds, numps = U.shape 
	H = umps2home(S, U)
	T = np.zeros(H.shape, dtype=np.int32)

	for t in xrange(1,nrounds):
		origin = H[t-1,:]
		destination = H[t,:]

		T[t,:] = D[origin-1,destination-1]		
	return T

def umps2cartesian(U1, U2):
	'''

		umps2cartesian computes the cartesean product of U1, U2 (ex: SQL CROSS JOIN) 
		
	'''	
	t, numps = U1.shape	
	
	nrounds   = t+U2.shape[0]
	

	CP  = np.zeros((nrounds,numps*numps),dtype=np.int32)

	for u1 in xrange(numps):
		for u2 in xrange(numps):
			col = u1*numps + u2			
			CP[:t,col] = U1[:,u1] 
			CP[t:,col] = U2[:,u2] 
	return CP