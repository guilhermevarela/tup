import numpy as np 

def umps2home(S, U):
	'''

		umps2homevenues hometeams / homevenues given an umps assignment

	'''	
	# import code; code.interact(local=dict(globals(), **locals()))
	#Home venues within schedule
	nrounds, nseries = U.shape 
	H = S[:,:,0]
	# HU = np.zeros(U.shape,dtype=np.int32)	
	#Performs home venus adjusted to umps assignment
	# for r in xrange(U.shape[0]):		
	# 	idr = U[r,:]-1
	# 	HU[r,:] = H[r,idr]
	# for t in xrange(H.shape[0]):
	# 	for u, g in enumerate(U[t,:]):
	# 		homeindex = H[t,g-1]-1
	# 		HU[t, u] = homeindex
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
	# AU = np.zeros(U.shape,dtype=np.int32)	
	
	#Performs home venue adjusted to umps assignment
	# for t in xrange(A.shape[0]):
	# 	for u, g in enumerate(U[t,:]):
	# 		advindex = A[t,g-1]-1
	# 		AU[t, u] =advindex
	rows = np.tile(np.arange(nrounds).reshape((nrounds,1)),(1,nseries))
	cols = U-1
	return A[rows,cols]

def umps2violations3(S,U):
	'''
		umps2violations3 every umpire sees a team at least once at home

	'''	
	H = umps2home(S, U)		
	nrounds, numps, _ =  S.shape
	nteams = 2*numps 
	V3 = np.zeros(H.shape, dtype=np.int32)
	aux = np.zeros((numps*2, U.shape[1]))
	umpsindex = np.arange(U.shape[1])


	for t in xrange(1,nrounds):
		for u, g in enumerate(U[t,:]):
			homeindex = H[t,g-1]-1
			aux[homeindex, u] +=1

	V3[-1,:]= (aux == 0).sum(axis=0) 		
	return V3 		

def umps2violations4(S,U,d1):
	'''

		umps2violations4 no umpire is in home site more then once in numps-d1 periods

	'''	
	nrounds, nseries = U.shape  
	numps = S.shape[1]

	H = umps2home(S,U)	
	V4 = np.zeros(H.shape, dtype=np.int32)

	for t in xrange(1,nrounds):
		y = max(t-(numps-d1),0)
		s = slice(y,t)
		for n in xrange(nseries):
			u = int(n/numps)
			v4 =  (H[y,u] == H[t,u]).sum() 		
			V4[t,n] =  v4
	return V4 		

def umps2violations5(S,U,d2):
	'''

		umps2violations5 no umpire sees a team more then once in int(numps/2)-d2 consecutive slots

	'''	
	nrounds, nseries = U.shape  
	numps=  S.shape[1] 
	H = umps2home(S,U)
	A = umps2adversaries(S,U)	
	
	V5 = np.zeros(U.shape, dtype=np.int32)
	for t in xrange(1,nrounds):
		y = max(t-(int(numps/2)-d2),0)
		s = slice(y,t)
		for n in xrange(nseries):
			u = int(n/numps)
			home =  (H[s,u] == H[t,u]).sum() +  (A[s,u] == H[t,u]).sum()
			adv  =  (H[s,u] == A[t,u]).sum() +  (A[s,u] == A[t,u]).sum()
			V5[t,n] =  home + adv 
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