import numpy as np

## dont forget to use MAP estimates
## prior estimate parameters

e = 0
D = 10
W = np.identity(D)
V = 1



class EMGMM:
	"estimating the parameters of a gaussian mixture model using em"

	def __init__(self, numComps):
		"initializes all the posterior parameters"
		self.numComps = numComps

		self.pi = np.ones(numComps)/numComps
		# initialize the mus with kmeans clustering
		self.mu = 
		self.sigmainverse = 

	def eStep(self, data):
		numComps = self.numComps
		numData = len(data)
		self.n = numData
		self.t = np.zeros(shape=(numData, numComps))
		
		for i in xrange(numData):
			for j in xrange(numComps):
				C = np.sum()
				self.t[i,j] = (self.pi[j]*normal(data[i], self.mu[j], self.sigma[j]))/C

		return likelihood

	def mStep(self, data):
		N = np.sum(np.sum(self.t, axis=0))
		self.N = N
		n = self.n
		for j in xrange(numComps):
			self.pi[j] = ((alpha[j]/numComps) + np.sum(self.t[:,j]))/(np.sum(alpha/numComps) + N)
			self.mu[j] = np.sum(self.t[:,j]*data)/(np.sum(self.t[:,j]) + epsilon[j])
			self.sigmainverse[j] = (np.linalg.inv(W) + np.sum(self.t[:,j]*np.outer((data-self.mu[j]),(data-self.mu[j]))))/((V - size(W)[0] -1)  + np.sum(self.t[:,j])
			self.sigma[j] = np.linalg.inv(self.sigmainverse)



	def estimateEM(self, data, numIters):
		"estimates posterior using the MAP solution to traditional EM"

		# set parameters
		epsilon = self.epsilon
		k = self.numComps
		d = self.dim
		n = len(data)

		# set hyperparameters of priors
		e = self.e
		w = self. 
		v = self.scalar


		likelihood = []
		for i in xrange(numIters):

			# do E step; this calculates P(Z|X,theta)
			likelihood.append(self.eStep(data))

			# if has converged, then stop
			if abs(likelihood[-1] - likelihood[-2]) < epsilon

			# do M step
			self.mStep(data)

	def estimateVariationalBayes(self, data, numIters):
		"esimtates posterior using variational bayes"




