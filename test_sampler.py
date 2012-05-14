import unittest
import numpy as np
from scipy.stats import norm
from numpy.random import normal,rand
import sampler
import matplotlib.pyplot as plot
import matplotlib.mlab as mlab
class SamplerTest(unittest.TestCase):

    def test_make_dist(self):
        test = rand(2,2)
        mc_sampler = sampler.Sampler(norm.pdf)
        f = mc_sampler.make_dist(test)
        self.assertEqual( test[0,0],f([0,0]) )
        self.assertEqual(test[0,1],f(np.array([0,1])))
        self.assertEqual(test[1,0],f(np.array([1,0])))
        self.assertEqual(test[1,1],f(np.array([1,1])))

    # STILL TO TEST: multivariate gaussian
    # test multivariate by getting a few answers from R
    def test_gauss_pdf(self):
#        test_point = normal()
        true_result = norm.pdf(0)
    
        for i in range(5):
            test_point = normal()
            true_result = norm.pdf(test_point)
            mc_sampler = sampler.Sampler(norm.pdf)
            possible = mc_sampler.gauss_pdf(np.array([test_point]),np.zeros(1),np.eye(1))
            self.assertEqual(format(true_result,'.4f'),format(possible,'.4f'))



    # test a 2x2 distribution
    # draw samples - calculate empirical expectation
    # should match 2x2 distributoin within reason
#    def test_sample_discrete(self): 
#       pass

    # test by passing it a standard normal gaussian
    # draw N samples from it
    # plot histogram of samples it should a gaussian
    def test_sample_continuous(self):
        # start with gaussian
        mu = 0
        sigma = 1
        mc_sampler = sampler.Sampler(norm.pdf)
        results = mc_sampler.sample(20000)
        hist_res = [x[0] for x in results]
        print hist_res
#        print results
        print np.mean(hist_res)
        print np.std(hist_res)

     #   n, bins, patches = plot.hist(results,30,normed=1)
     #   print bins
#        print n
    #    y = mlab.normpdf( bins, mu, sigma)
    #    l = plot.plot(bins, y, 'r--', linewidth=1)
    #    plot.show()
        #plot.savefig('test_sampler.png')

#    def emp_exp(self,samples):
#        exp = np.zeros((len(samples[0],len(samples[0]))))
#        for sample in samples:
#            for i in range(len(sample)):
#                pass
#
#    def test_sample_discrete(self):
#        dist = [ [.36, .25],[.25, .14]]
#        mc_sampler = sampler.Sampler(dist)
#        print dist
#        results = mc_sampler.sample(1000)
#        print results

if __name__ == '__main__':
    unittest.main()