import scipy as sp
import numpy as np
from numpy.random import multivariate_normal as gaussian
from scipy.stats import norm

class Sampler:

	#assumptions - if continuous returns either a float or Nx1 array 
	#to do - error checking 
	def __init__(self,dist,t="cont"):
		if type == "cont":
			self.dist = dist
			self.t = t
			self.shape = np.array(self.dist())

#		self.dist = np.array(dist)
#		self.norm_dist = normalize(dist)
#		self.shape = dist.shape


	def normalize(self,dist):
		normalized_dist = np.sum(dist)
		if normalized_dist > 0:
			return np.divide(dist,norm)
		# if not raise error to do

	def cont_sample(self,num_samples,burn_in):
		samples = []
		cov = np.eye(self.shape)
		sample = gaussian(np.zeros(self.shape),cov)
		#i'm going to use dot product because I know that works in the scalar case
		#I believe it works in the vector space but I'm not sure - driving me crazy
		for i in range(num_samples):
			proposal = gaussian(sample,cov)
			proposal_prob = min(1, np.log(np.exp(
											np.dot(self.dist(proposal),self.dist(sample)) - 
											np.dot(norm(),norm() ) //placeholders
											)
										)
								)


	def sample(self,num_samples,burn_in=0):
		samples = []
		sample = gaussian(np.zeros(self.shape[0]),np.eye(self.shape[0]))
		for i in range(num_samples):
			pass



