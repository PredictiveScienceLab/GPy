# Written by Ilias Bilionis (ibilion@purdue.edu)
"""
Test sampling the GP with MCMC.
"""


import unittest
import numpy as np
import GPy
import pymc as pm


class PyMCTestCase(unittest.TestCase):

    def test_olympic_100m_men(self):
        gpy_model = GPy.examples.regression.toy_rbf_1d(plot=False, optimize=False)
        gpy_model.likelihood.variance.set_prior(GPy.priors.Jeffreys())
        gpy_model.kern.variance.set_prior(GPy.priors.Jeffreys())
        gpy_model.kern.lengthscale.set_prior(GPy.priors.LogLogistic())
        gpy_model.pymc_db = 'pickle'
        gpy_model.pymc_db_opts['dbname'] = 'foo.pcl'
        gpy_model.pymc_mcmc.sample(10000, burn=1000, thin=10)
        theta = gpy_model.pymc_mcmc.trace('hyperparameters')[:]
        phi = gpy_model.pymc_mcmc.trace('transformed_hyperparameters')[:]
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot(phi)
        for i in xrange(phi.shape[1]):
            fig, ax = plt.subplots()
            ax.hist(theta[:, i])
        gpy_model.plot()
        plt.show(block=True)


if __name__ == '__main__':
    unittest.main()
