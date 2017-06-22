import numpy as np 

def umps2home(S, U):
	'''

		umps2homevenues hometeams / homevenues given an umps assignment

	'''	
	#Home venues within schedule
	H = S[:,:,0]
	HU = np.zeros(H.shape,dtype=np.int32)	
	#Performs home venus adjusted to umps assignment
	for r in xrange(H.shape[0]):		
		HU[r,:] = H[r,U[r,:]-1]
	return HU 

def umps2adversaries(S, U):
	'''

		umps2adversaries returns the adversary team an umps assignent

	'''	
	#adversary team within schedule
	A = S[:,:,1]
	AU = np.zeros(A.shape,dtype=np.int32)	
	
	#Performs home venus adjusted to umps assignment
	for r in xrange(A.shape[0]):
		AU[r,:] = A[r,U[r,:]-1]
	return AU	

def umps2violations3(S,U):
	'''
		umps2violations3 every umpire sees a team at least once at home

	'''	
	H = umps2home(S,U)
	nrounds, numps =  H.shape
	V3 = np.zeros(H.shape, dtype=np.int32)
	aux = np.zeros((numps*2,numps))
	umpsindex = np.arange(numps)

	for t in xrange(1,nrounds):
		homeindex = H[t,:]-1
		aux[homeindex, umpsindex] +=1

	V3[-1,:]= (aux == 0).sum(axis=0).T 		
	return V3 		

def umps2violations4(S,U,d1):
	'''

		umps2violations4 no umpire is in home site more then once in numps-d1 periods

	'''	
	H = umps2home(S,U)
	nrounds, numps =  H.shape
	V4 = np.zeros(H.shape, dtype=np.int32)

	for t in xrange(1,nrounds):
		y = max(t-(numps-d1),0)
		s = slice(y,t)
		for u in xrange(numps):
			v4 =  (H[y,u] == H[t,u]).sum() 		
			V4[t,u] =  v4
	return V4 		

def umps2violations5(S,U,d2):
	'''

		umps2violations5 no umpire sees a team more then once in int(numps/2)-d2 consecutive slots

	'''	
	H = umps2home(S,U)
	A = umps2adversaries(S,U)
	nrounds, numps =  H.shape
	V5 = np.zeros(H.shape, dtype=np.int32)

	for t in xrange(1,nrounds):
		y = max(t-(int(numps/2)-d2),0)
		s = slice(y,t)
		for u in xrange(numps):

			home =  (H[s,u] == H[t,u]).sum() +  (A[s,u] == H[t,u]).sum()
			adv  =  (H[s,u] == A[t,u]).sum() +  (A[s,u] == A[t,u]).sum()
			V5[t,u] =  home + adv 
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