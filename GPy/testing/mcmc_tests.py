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
        gpy_model = GPy.examples.regression.olympic_100m_men(plot=False, optimize=False)
        gpy_model._X_predict=np.linspace(1850., 2050., 100)[:, None]
        gpy_model.likelihood.variance.set_prior(GPy.priors.Jeffreys())
        gpy_model.kern.variance.set_prior(GPy.priors.Jeffreys())
        gpy_model.kern.lengthscale.set_prior(GPy.priors.Jeffreys())
        gpy_model.pymc_db = 'hdf5'
        gpy_model.pymc_db_opts['dbname'] = 'foo.h5'
        gpy_model.pymc_mcmc.sample(100000, burn=10000, thin=1000)
        theta = gpy_model.pymc_mcmc.trace('hyperparameters')[:]
        phi = gpy_model.pymc_mcmc.trace('transformed_hyperparameters')[:]
        means = gpy_model.pymc_mcmc.trace('predictive_mean')[:]
        posterior_samples = gpy_model.pymc_mcmc.trace('posterior_samples')[:]
        posterior_samples = np.vstack(posterior_samples)
        p_05 = np.percentile(posterior_samples, 5, axis=0)
        p_95 = np.percentile(posterior_samples, 95, axis=0) 
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot(phi)
        for i in xrange(phi.shape[1]):
            fig, ax = plt.subplots()
            ax.hist(theta[:, i])
        fig, ax = plt.subplots()
        #gpy_model.plot(ax=ax)
        for i in xrange(means.shape[0]):
            ax.plot(gpy_model.X_predict, means[i, :], 'r', linewidth=0.5)
        ax.plot(gpy_model.X, gpy_model.Y, 'x')
        ax.fill_between(gpy_model.X_predict.flatten(), p_05, p_95, color='red',
                        alpha=0.25)
        plt.show(block=True)


if __name__ == '__main__':
    unittest.main()
