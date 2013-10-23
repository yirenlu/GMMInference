from __future__ import division
import numpy as np
import math
import scipy
import scipy.sparse as sp
import scipy.sparse.linalg as spln
from sklearn import mixture
import matplotlib.pyplot as plt
import Pycluster as pc

dim = 5
numComps = 2

class EMGMM:
	"estimating the parameters of a gaussian mixture model using em"

	def __init__(self, numComps=None, dim=5, data=None, epsilon=math.pow(10, -10), wishartScalar=1, wishartScale=np.identity(dim), dirichlet=np.ones(numComps), normalMu=0, normalSigma=np.identity(dim)):
		
		# INITIALIZE ALL POSTERIOR PARAMETERS

		self.d = dim
		self.k = numComps
		self.n = len(data)

		# INITIALIZE ALL PRIOR PARAMETERS

		self.e = normalSigma
		self.m = normalMu
		self.w = wishartScale
		self.v = wishartScalar
		self.di = dirichlet
		self.epsilon = epsilon

		# INITIALIZE ALL PRIORS USING k-means CLUSTERING

		# INITIALIZE THE MUS

		labels, error, nfound = pc.kcluster(data, self.k)#, iter=300, thresh=1e-05)
	
		centroids, _ = pc.clustercentroids(data, clusterid=labels)

		self.mu = centroids

		self.pointsInComp = [[] for comp in xrange(self.k)]
		for n in xrange(self.n):
			self.pointsInComp[labels[n]].append(data[n])

		# INITIALIZE THE COVARIANCE MATRIX
		self.sigma = [np.cov(np.array(kpoints).T) for kpoints in self.pointsInComp]

		# INITIALIZE THE WEIGHTS
		self.pi = [len(l)/data.shape[0] for l in self.pointsInComp]

	def normal(self, x, mean, covariance):

		# THIS EVALUATES THE MULTIVARIATE NORMAL DISTRIBUTION WITH MEAN AND COVARIANCE AS SPECIFIED
		nx = len(covariance)
		tmp = -0.5*(nx*math.log(2*math.pi)+np.linalg.slogdet(covariance)[1])
		err = x-mean

		if (sp.issparse(covariance)):
			numerator = spln.spsolve(covariance, err).T.dot(err)
		else:
			numerator = np.linalg.solve(covariance, err).T.dot(err)

		return math.exp(tmp-numerator)


	def eStep(self, data):
		
		self.t = np.zeros(shape=(self.n, self.k))
		self.C = np.zeros(shape=(self.n, 1))
		
		for l in xrange(self.n):
			for m in xrange(self.k):
				self.C[l] += np.sum(self.pi[m]*self.normal(data[l], self.mu[m], self.sigma[m]))

		for i in xrange(self.n):
			for j in xrange(self.k):
				self.t[i,j] = (self.pi[j]*self.normal(data[i], self.mu[j], self.sigma[j]))/np.float(self.C[i])

		likelihood = np.sum(np.log(self.C))
		return likelihood

	def mStep(self, data):
		N = np.sum(np.sum(self.t, axis=0))
		self.N = N
		n = self.n
		sigmainverse = []
		for w in xrange(self.k):
			sigmainverse.append(np.linalg.inv(self.sigma[w]))

		for j in xrange(numComps):
			self.pi[j] = ((self.di[j]/numComps) + np.sum(self.t[:,j]))/(np.sum(self.di/numComps) + N)
			self.mu[j] = np.dot(self.t[:,j].T, data)/(np.sum(self.t[:,j]) + self.e[j])
			outerProduct = [self.t[n,j]*np.outer((data[n]-self.mu[j]), (data[n]-self.mu[j])) for n in xrange(self.n)]
			sigmainverse[j] = (np.linalg.inv(self.w) + np.sum(outerProduct))/((self.v - self.w.shape[0] -1)  + np.sum(self.t[:,j]))
			self.sigma[j] = np.linalg.inv(sigmainverse[j])
			

	def estimateEM(self, data, numIters):
		"estimates posterior using the MAP solution to traditional EM"

		likelihood = []
		for i in xrange(numIters):

			# do E step; this calculates P(Z|X,theta)
			likelihood.append(self.eStep(data))

			# if has converged, then stop
			if i > 0 and abs(likelihood[-1] - likelihood[-2]) < self.epsilon:
				print self.mu
				print self.pi
				print self.sigma
				break

			# do M step
			self.mStep(data)

def main():
	obs = np.concatenate((np.random.randn(100, 5), 10 + np.random.randn(300, 5)))
	plt.plot(obs, 'ro')
	plt.show()
	g = EMGMM(2,5,obs)
	g.estimateEM(obs,100)


if __name__ == "__main__":
	main()






