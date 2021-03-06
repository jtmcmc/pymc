import scipy as sp
import numpy as np
from numpy.random import multivariate_normal as gaussian
from numpy.linalg import inv,slogdet
from math import pi
from scipy.stats import norm
import types
#todo - make tabs 4 spaces this looks terrible
#todo - MAKE TESTS
#todo - make it accept either an array or a function
#todo - MCMCMC 
# really I should be doing everythign in log space for numerical stability
# also abstract the array thing out check if dist is object or array
# if array make a normalized anonymous function
class Sampler:

    #assumptions - if continuous returns either a float or Nx1 array 
    #to do - error checking 
    def __init__(self,dist,size=1,t="cont"):
        if t == 'discrete':
            self.dist= self.make_dist(dist)
            print  
        if t == 'cont':
            self.dist = dist
        else:
            raise TyepeError("'t accepts 'cont' or 'discrete'")
        self.t = t
        self.size = size

#self.dist = np.array(dist)
#self.norm_dist = normalize(dist)
#self.shape = dist.shape
#def to test 
    def make_dist(self,arr):
        return lambda x: np.array(arr).item(tuple(x))

    def normalize(self,dist):
        normalization_const = np.sum(dist)
        if normalized_dist > 0:
            return np.divide(dist,norm)
        # if not raise error to do

    #not currently checking if COV is PSD
    #this will probably choke on a larger 
    #expects that point / mean / cov are appropriate arrays
    def gauss_pdf(self,point,mean,cov):
        k = len(point)
        sign,logdet = slogdet(cov)
        logdet = sign*logdet 
        return np.exp(-.5*np.dot( 
                    np.dot(
                        np.transpose(point - mean),inv(cov)),
                        (point-mean)						
                    ) - (k/2.0)*np.log(2.0*pi) + .5*logdet   
                )
    #we are assuming proposal distribution is always a standard gaussian
    #could allow for cov to be set optionally
    def sample(self,num_samples,burn_in=0):
        samples = []
        cov = np.eye(self.size)
        sample = gaussian(np.zeros(self.size),cov)

        #i'm going to use dot product because I know that works in the scalar case
        #I believe it works in the vector space but I'm not sure - driving me crazy
        while len(samples) < num_samples:
            proposal = gaussian(sample,cov)
            
            proposal_prob = min(1, np.exp(np.log(self.dist(proposal)) + 
                                    np.log(self.gauss_pdf(sample,proposal,cov)) -
                                    np.log(self.dist(sample)) -
                                    np.log(self.gauss_pdf(proposal,sample,cov))
                                    )
                                )
            if proposal_prob > np.random.rand():
                samples.append(sample)
                sample = proposal
        return samples



