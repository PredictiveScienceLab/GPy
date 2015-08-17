# Written by Ilias Bilionis (ibilion@purdue.edu)
"""
Test sampling the GP with MCMC.
"""


import unittest
import numpy as np
import GPy
import pymc as pm
from scipy.misc import logsumexp
from scipy.integrate import quad


class TestModel(GPy.Model):
    """
    A simple test model.
    """

    def __init__(self, x=1.):
        GPy.core.Model.__init__(self, 'TestModel')
        xp = GPy.core.Param('x', x)
        self.link_parameter(xp)

    def log_likelihood(self):
        return -0.5 * (self.x - 3.) ** 2.


class Mixture(GPy.priors.Prior):
    """
    A prior representing the mixture of two Gaussians.
    """

    def __init__(self, mu1=0.5, sigma1=0.2, mu2=1.3, sigma2=0.4,
                 w=0.5):
        self.left = GPy.priors.LogGaussian(mu=mu1, sigma=sigma1)
        self.right = GPy.priors.LogGaussian(mu=mu2, sigma=sigma2)
        self.logw1 = np.log(w)
        self.logw2 = np.log(1. - w)
    
    def lnpdf(self, x):
        log_ps = np.array([self.logw1 + self.left.lnpdf(x),
                           self.logw2 + self.right.lnpdf(x)])
        if log_ps.shape[1] == 1:
            return logsumexp(log_ps)
        return logsumexp(log_ps, axis=0)

    def __str__(self):
        return 'Mixture'


class Rosenbrock(GPy.priors.Prior):
    """
    A prior representing Rosenbrock's density.
    """
    
    def __init__(self, k=1./20.):
        self.k = k

    def lnpdf(self, x):
        return -self.k * (100. * (x[1] - x[0] ** 2) ** 2 + (1. - x[0]) ** 2)

    def __str__(self):
        return 'Rosenbrock'


class PyMCTestCase(unittest.TestCase):

    #def test_sampling_from_1d_dist(self):
    #    """
    #    Test if we can actually sample from a 1D distribution.
    #    """
    #    m = TestModel()
    #    prior = GPy.priors.LogGaussian(.1, 0.9)
    #    m.x.set_prior(prior)
    #    m.x.unconstrain()
    #    m.x.constrain(GPy.constraints.Logexp())
    #    f = lambda(x): np.exp(-m._objective(x))
    #    # Now use the adaptive metropolis algorithms to sample from this thing
    #    m.pymc_mcmc.sample(100000, burn=1000, thin=100) 
    #    theta = m.pymc_mcmc.trace('hyperparameters')[:]
    #    phi = m.pymc_mcmc.trace('transformed_hyperparameters')[:]
    #    thetas = np.linspace(theta.min(), theta.max(), 100)
    #    result = quad(f, -np.inf, np.inf)
    #    print m.get_log_evidence_history()
    #    print result
    #    quit()
    #    # UNCOMMENT TO SEE GRAPHICAL COMPARISON
    #    #import matplotlib.pyplot as plt
    #    #fig, ax = plt.subplots()
    #    #ax.hist(theta, bins=100, alpha=0.5, normed=True)
    #    #true_pdf = prior.pdf(thetas)
    #    #ax.plot(thetas, true_pdf, 'r', linewidth=2)
    #    plt.show(block=True)

    #def test_sampling_from_1d_mixture_dist(self):
    #    """
    #    Test if sampling works for distributions with two modes.
    #    """
    #    m = TestModel()
    #    prior = Mixture()
    #    m.x.set_prior(prior)
    #    m.x.unconstrain()
    #    m.x.constrain(GPy.constraints.Logexp())
    #    m.pymc_mcmc.sample(10000, burn=1000, thin=10)
    #    theta = m.pymc_mcmc.trace('hyperparameters')[:]
    #    # UNCOMMENT TO SEE GRAPH
    #    #import matplotlib.pyplot as plt
    #    #fig, ax = plt.subplots()
    #    #ax.hist(theta, bins=100, alpha=0.5, normed=True)
    #    #thetas = np.linspace(theta.min(), theta.max(), 100)
    #    #true_pdf = prior.pdf(thetas)
    #    #ax.plot(thetas, true_pdf, 'r', linewidth=2)
    #    #plt.show(block=True)

    #def test_rosenbrock_density(self):
    #    """
    #    Test the Rosenbrock (1960) density which looks like a banana.
    #    """
    #    m = TestModel(x=[5, 10.])
    #    prior = Rosenbrock()
    #    m.x.set_prior(prior)
    #    m.x.unconstrain()
    #    m.pymc_mcmc.sample(10000, burn=1000, thin=100)
    #    theta = m.pymc_mcmc.trace('hyperparameters')[:]
    #    # UNCOMMENT TO SEE GRAPH
    #    #import matplotlib.pyplot as plt
    #    #fig, ax = plt.subplots()
    #    #ax.plot(theta[:, 0], theta[:, 1], '.')
    #    #plt.show(block=True)

    def test_olympic_100m_men(self):
        np.random.seed(12345)
        gpy_model = GPy.examples.regression.olympic_100m_men(plot=False, optimize=True)
        gpy_model._X_predict=np.linspace(1850., 2050., 100)[:, None]
        gpy_model.update_model(False)
        gpy_model.likelihood.variance.set_prior(GPy.priors.Jeffreys())
        gpy_model.kern.variance.set_prior(GPy.priors.Jeffreys())
        #gpy_model.kern.lengthscale.set_prior(GPy.priors.Jeffreys())
        gpy_model.kern.lengthscale.set_prior(GPy.priors.LogGaussian(mu=500., sigma=100.))
        gpy_model.update_model(True)
        gpy_model.pymc_step_method_params['verbose'] = 0
        gpy_model.pymc_step_method_params['shrink_if_necessary'] = True

        # Trace the following QoI over the MCMC chain
        gpy_model.pymc_trace_denoised_max()
        gpy_model.pymc_trace_denoised_argmax()
        gpy_model.pymc_trace_posterior_samples()
        gpy_model.pymc_trace_expected_improvement(denoised=True)

        # Run the MCMC chain
        gpy_model.pymc_mcmc.sample(1000, burn=100, thin=10,
                                   tune_throughout=False, verbose=False)

        # Plot the evidence even though it is not accurate
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        log_E  = gpy_model.get_log_evidence_history()
        idx = np.arange(log_E.shape[0])
        plt.plot(idx, log_E)

        # Get all the QoI's that we have been tracing
        theta = gpy_model.pymc_mcmc.trace('hyperparameters')[:]
        phi = gpy_model.pymc_mcmc.trace('transformed_hyperparameters')[:]
        means = gpy_model.pymc_mcmc.trace('predictive_mean')[:]
        posterior_samples = gpy_model.pymc_mcmc.trace('posterior_samples')[:]
        posterior_samples = np.vstack(posterior_samples)
        denoised_posterior_samples = \
                gpy_model.pymc_mcmc.trace('denoised_posterior_samples')[:]
        denoised_posterior_samples = np.vstack(denoised_posterior_samples)
        posterior_max = gpy_model.pymc_mcmc.trace('denoised_max')[:]
        posterior_max = np.hstack(posterior_max)
        posterior_argmax = gpy_model.pymc_mcmc.trace('denoised_argmax')[:]
        posterior_argmax = np.hstack(posterior_max)
        log_like = gpy_model.pymc_mcmc.trace('log_likelihood')[:]
        log_prior = gpy_model.pymc_mcmc.trace('log_prior')[:]
        ei_all = gpy_model.pymc_mcmc.trace('denoised_ei_min')[:]
        ei = ei_all.mean(axis=0)

        # Plot evolution of transformed hyperparameters
        fig, ax = plt.subplots()
        ax.plot(theta[:, 0], theta[:, 1], '.')
    
        # Plot evolution of probability of MCMC samples
        fig, ax = plt.subplots()
        ax.plot(log_like + log_prior)

        # Plot the histograms of the parameters of the model
        fig, ax = plt.subplots()
        ax.plot(phi)
        for i in xrange(phi.shape[1]):
            fig, ax = plt.subplots()
            ax.hist(theta[:, i])

        # Plot the predictions
        fig, ax = plt.subplots()
        # Predictive mean of each point at MCMC chain
        for i in xrange(means.shape[0]):
            ax.plot(gpy_model.X_predict, means[i, :], 'r', linewidth=0.5)
        # Observed data
        ax.plot(gpy_model.X, gpy_model.Y, 'x')
        # 95% predictive error bars including noise
        p_025 = np.percentile(posterior_samples, 2.5, axis=0)
        p_975 = np.percentile(posterior_samples, 97.5, axis=0) 
        ax.fill_between(gpy_model.X_predict.flatten(), p_025, p_975,
                        color='red', alpha=0.25)
        # 95% predictive error bars without noise
        dnp_025 = np.percentile(denoised_posterior_samples, 2.5, axis=0)
        dnp_975 = np.percentile(denoised_posterior_samples, 97.5, axis=0) 
        ax.fill_between(gpy_model.X_predict.flatten(), dnp_025, dnp_975,
                        color='blue', alpha=0.25)
        # Get a twin axis and plot the expected improvement on it
        ax2 = ax.twinx()
        ax2.plot(gpy_model.X_predict, ei, 'g', linewidth=2)
        ax2.plot(gpy_model.X_predict, ei_all.T, 'g', linewidth=0.1)
        plt.setp(ax2.get_yticklabels(), 'color', 'g')

        # Plotting the raw posterior samples
        fig, ax = plt.subplots()
        ax.plot(gpy_model.X_predict, posterior_samples.T, 'r', linewidth=0.1)

        # Plot the denoised posterior samples
        fig, ax = plt.subplots()
        ax.plot(gpy_model.X_predict, denoised_posterior_samples.T, 'r', linewidth=0.1)

        # Plot the histogram of the location of the maximum
        X_max = gpy_model.X_predict[posterior_argmax.astype('int'), 0]
        fig, ax = plt.subplots()
        ax.hist(X_max, normed=True, alpha=0.25)
        # Plot the histogram of the value of the maximum
        fig, ax = plt.subplots()
        ax.hist(posterior_max, color='red', alpha=0.25)
        plt.show(block=True)


if __name__ == '__main__':
    unittest.main()
