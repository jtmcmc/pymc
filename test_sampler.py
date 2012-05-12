import unittest
import numpy as np
from scipy.stats import norm
from numpy.random import normal
import sampler

class SamplerTest(unittest.TestCase):

    def setUp(self):
        self.test = 0

#    def test_make_dist(self):
#        pass

    # test univariate condition by using the norm function
    # test multivariate by getting a few answers from R
    def test_gauss_pdf(self):
#        test_point = normal()
        true_result = norm.pdf(0)
        mc_sampler = sampler.Sampler(norm.pdf)
        possible = mc_sampler.gauss_pdf(np.array([0]),np.zeros(1),np.eye(1))
        print possible
        self.assertEqual(true_result,possible)

    # test a 2x2 distribution
    # draw samples - calculate empirical expectation
    # should match 2x2 distributoin within reason
 #   def test_sample_discrete(self): 
 #       pass

    # test by passing it a standard normal gaussian
    # draw N samples from it
    # plot histogram of samples it should a gaussian
  #  def test_sample_continuous(self):
  #      pass

if __name__ == '__main__':
    unittest.main()