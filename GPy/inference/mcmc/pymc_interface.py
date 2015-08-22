# Written by Ilias Bilionis (ibilion@purdue.edu)
"""
Methods that allow one to interface GPy with PyMC.

We keep it as simple as possible.
"""


import pymc as pm
import numpy as np
import math
from ei import expected_improvement
from scipy.misc import logsumexp
from scipy.stats import hmean


__all__ = ['PyMCInterface']



class PyMCInterface(object):

    """
    Defines a PyMC interface for ``GPy.core.model.Model``.

    Parameters
    ----------

        - pymc_db:       string
                         The name of the database backend that will store the values of everything
                         sampled by the MCMC loop. Existing databases are: `ram, no_trace, pickle, txt, sqlite, hdf5`.
                         See this: https://pymc-devs.github.io/pymc/database.html.
                         The default is RAM.
        - pymc_db_opts:  dictionary
                         Options for the database. These are as folllows:
                            + ram:
                                no options
                            + no_trace:
                                no options
                            + pickle:
                                ~ dbname:   name of output file
                            + txt:
                                ~ dbname:   name of output file
                            + sqlite:
                                ~ dbname:   name of output file
                            + hdf5:
                                ~ dbname:       name of output file
                                ~ dbmode:       file mode, 'a', 'w', or 'r'
                                ~ dbcomplevel:  compression level
                                ~ dbcomplib:    compression library
    """

    # Holder for the PyMC model dictionary
    _pymc_model = None

    # Holder for the PyMC MCMC sampler
    _pymc_mcmc = None

    # The type of database we want to use in for saving the results of MCMC
    _pymc_db = None

    # The options for the database
    _pymc_db_opts = None
    
    # The step method you want to assign to sample the hyper-parameters
    _pymc_step_method = None
    
    # The parameters you want to pass to the step method class
    _pymc_step_method_params = None

    # Points on which to draw predictions
    _X_predict = None

    # Number of samples to take for predictions
    _num_predict = None

    # A list of quantities you would like to compute at each MCMC sample
    _deterministics = None

    @property
    def pymc_db(self):
        return self._pymc_db

    @pymc_db.setter
    def pymc_db(self, value):
        assert value in ['ram', 'no_trace', 'pickle', 'txt', 'sqlite', 'hdf5']
        self._pymc_db = value

    @property
    def pymc_db_opts(self):
        return self._pymc_db_opts

    @pymc_db_opts.setter
    def pymc_db_opts(self, value):
        assert isinstance(value, dict)
        self._pymc_db_opts = value
        
    @property
    def pymc_step_method(self):
        return self._pymc_step_method
        
    @property
    def pymc_step_method_params(self):
        return self._pymc_step_method_params

    @property
    def X_predict(self):
        return self._X_predict

    @property
    def num_predict(self):
        return self._num_predict

    @property
    def pymc_deterministics(self):
        return self._deterministics

    def __init__(self, pymc_db='ram', pymc_db_opts={},
                 pymc_step_method=pm.AdaptiveMetropolis,
                 pymc_step_method_params={},
                 X_predict=None, num_predict=10):
        self._pymc_db = pymc_db
        self._pymc_db_opts = pymc_db_opts
        self._pymc_step_method = pymc_step_method
        self._pymc_step_method_params = pymc_step_method_params
        self._X_predict = X_predict
        self._num_predict = num_predict
        self._deterministics = []
        self.has_posterior_samples = False
        self.has_denoised_posterior_samples = False

    def pymc_trace_deterministic(self,
                               func,
                               name,
                               doc='PyMC deterministic',
                               parents=['predictive_mean',
                                        'predictive_covariance',
                                        'hyperparameters'],
                               trace=True,
                               plot=False,
                               **kwargs):
        """
        Add a deterministic to be monitored by the class at each MCMC step.

        :param func:    The function to be evaluated.
        :param name:    The name of the deterministic.
        :param doc:     Brief description of what the deterministic is.
        :param parents: The names of the inputs of ``func``.
        :param trace:   ``True`` if you want to save it to the data base

        The rest of the keywords are the same as in ``pymc.Deterministic``.
        """
        self.pymc_deterministics.append({'name': name,
                                         'eval': func,
                                         'doc': doc,
                                         'parents': parents,
                                         'plot': plot,
                                         'trace': trace})
        self.pymc_deterministics[-1].update(kwargs)

    def pymc_trace_posterior_samples(self):
        """
        Adds the posterior samples deterministic.
        """
        if self.X_predict is not None:
            def func(predictive_mean, predictive_covariance, hyperparameters):
                return np.random.multivariate_normal(predictive_mean.flatten(),
                                                     predictive_covariance,
                                                     self.num_predict)
            self.pymc_trace_deterministic(func,
                                        'posterior_samples',
                                        dtype=np.ndarray)
            self.has_posterior_samples = True

    def pymc_trace_denoised_posterior_samples(self):
        """
        Adds the posterior samples deterministic.

        Be careful: This only work with a Gaussian likelihood and under the
        assumption that the variance of the likelihood is the last hyperparameter.

        Modify this accordingly if you have a non-standard GP model.
        """
        if self.X_predict is not None:
            def func(predictive_mean, predictive_covariance, hyperparameters):
                return np.random.multivariate_normal(predictive_mean.flatten(),
 predictive_covariance - hyperparameters[-1] * np.eye(predictive_covariance.shape[0]),
                                                     self.num_predict)
            self.pymc_trace_deterministic(func,
                                        'denoised_posterior_samples',
                                        dtype=np.ndarray)
            self.has_denoised_posterior_samples = True

    def pymc_trace_func_of_posterior_samples(self, func, name,
                                                         **kwargs):
        """
        Adds a deterministic that is a function of each one of the posterior
        samples.
        """
        if not self.has_posterior_samples:
            self.pymc_trace_posterior_samples()
        if self.has_posterior_samples:
            self.pymc_trace_deterministic(func, name,
                                          parents=['posterior_samples'],
                                          **kwargs)

    def pymc_trace_expected_improvement(self, mode='min', denoised=False, **kwargs):
        parents = ['predictive_mean', 'predictive_covariance']
        if not denoised:
            name = 'ei_' + mode
            def func(predictive_mean, predictive_covariance):
                    return expected_improvement(predictive_mean,
                                         np.diag(predictive_covariance),
                                         self.Y, mode=mode, noise=0.)
        else:
            name = 'denoised_ei_' + mode
            def func(predictive_mean, predictive_covariance,
                          denoised_outputs, hyperparameters):
                    return expected_improvement(predictive_mean,
                                         np.diag(predictive_covariance),
                                         denoised_outputs, mode=mode,
                                         noise=hyperparameters[-1])
            parents.append('denoised_outputs')
            parents.append('hyperparameters')
        self.pymc_trace_deterministic(func,
                                      name,
                                      dtype=float,
                                      parents=parents,
                                      **kwargs)

    def pymc_trace_max(self):
        """
        Adds a deterministic that traces the maximum of the posterior samples.
        """
        func = lambda(posterior_samples): np.max(posterior_samples, axis=1)
        self.pymc_trace_func_of_posterior_samples(func, 'max',
                                                   doc='Maximum over X_predict')

    def pymc_trace_argmax(self):
        func = lambda(posterior_samples): np.argmax(posterior_samples, axis=1)
        self.pymc_trace_func_of_posterior_samples(func, 'argmax',
                                                  doc='Argmax over X_predict')

    def pymc_trace_min(self):
        """
        Adds a deterministic that traces the minimum of the posterior samples.
        """
        func = lambda(posterior_samples): np.min(posterior_samples, axis=1)
        self.pymc_trace_func_of_posterior_samples(func, 'min',
                                                  doc='Minimum over X_predict')

    def pymc_trace_argmin(self):
        func = lambda(posterior_samples): np.argmin(posterior_samples, axis=1)
        self.pymc_trace_func_of_posterior_samples(func, 'argmin',
                                                  doc='Argmin over X_predict')

    def pymc_trace_func_of_denoised_posterior_samples(self, func, name,
                                                         **kwargs):
        """
        Adds a deterministic that is a function of each one of the posterior
        samples.
        """
        if not self.has_denoised_posterior_samples:
            self.pymc_trace_denoised_posterior_samples()
        if self.has_denoised_posterior_samples:
            self.pymc_trace_deterministic(func, name,
                                          parents=['denoised_posterior_samples'],
                                          **kwargs)

    def pymc_trace_denoised_max(self):
        """
        Adds a deterministic that traces the maximum of the posterior samples.
        """
        func = lambda(denoised_posterior_samples): np.max(denoised_posterior_samples, axis=1)
        self.pymc_trace_func_of_denoised_posterior_samples(func, 'denoised_max',
                                                   doc='Maximum over X_predict')

    def pymc_trace_denoised_argmax(self):
        func = lambda(denoised_posterior_samples): np.argmax(denoised_posterior_samples, axis=1)
        self.pymc_trace_func_of_denoised_posterior_samples(func, 'denoised_argmax',
                                                  doc='Argmax over X_predict')

    def pymc_trace_denoised_min(self):
        """
        Adds a deterministic that traces the minimum of the posterior samples.
        """
        func = lambda(denoised_posterior_samples): np.min(denoised_posterior_samples, axis=1)
        self.pymc_trace_func_of_denoised_posterior_samples(func, 'denoised_min',
                                                  doc='Minimum over X_predict')

    def pymc_trace_denoised_argmin(self):
        func = lambda(denoised_posterior_samples): np.argmin(denoised_posterior_samples, axis=1)
        self.pymc_trace_func_of_denoised_posterior_samples(func, 'denoised_argmin',
                                                  doc='Argmin over X_predict')

    @property
    def pymc_model(self):
        """
        Take the current model and turn it into a PyMC model.
        """
        if self._pymc_model is not None:
            return self._pymc_model
        @pm.stochastic(dtype=np.ndarray)
        def transformed_hyperparameters(value=self.optimizer_array):
            return 0.
        @pm.deterministic(trace=False)
        def model(theta=transformed_hyperparameters, obj=self):
            obj.optimizer_array = theta
            res = {}
            res['log_p'] = -obj.objective_function()
            res['log_like'] = obj.log_likelihood()
            res['log_prior'] = obj.log_prior()
            phi = np.ndarray(theta.shape)
            obj._inverse_hyperparameter_transform(theta, phi)
            res['theta'] = theta
            res['phi'] = phi
            if obj.X_predict is not None:
                tmp = obj.predict(obj.X_predict, full_cov=True)
                res['predictive_mean'] = tmp[0]
                res['predictive_covariance'] = tmp[1]
            # The projections of the observations which are useful in defining
            # the expected improvement for noisy cases
            res['denoised_outputs'] = self.predict(self.X)[0]
            return res
        @pm.stochastic(observed=True, trace=False)
        def observation(value=1., model=model):
            return model['log_p']
        @pm.deterministic(dtype=np.ndarray)
        def hyperparameters(model=model):
            return model['phi']
        @pm.deterministic(dtype=float)
        def log_likelihood(model=model):
            return model['log_like']
        @pm.deterministic(dtype=float)
        def log_prior(model=model):
            return model['log_prior']
        @pm.deterministic(dtype=np.ndarray)
        def denoised_outputs(model=model, trace=False):
            return model['denoised_outputs']
        pymc_model = {'transformed_hyperparameters': transformed_hyperparameters}
        pymc_model['hyperparameters'] = hyperparameters
        pymc_model['log_likelihood'] = log_likelihood
        pymc_model['log_prior'] = log_prior
        pymc_model['denoised_outputs'] = denoised_outputs
        if self.X_predict is not None:
            @pm.deterministic(dtype=np.ndarray)
            def predictive_mean(model=model):
                return model['predictive_mean']
            @pm.deterministic(dtype=np.ndarray)
            def predictive_covariance(model=model):
                return model['predictive_covariance']
            pymc_model['predictive_mean'] = predictive_mean
            pymc_model['predictive_covariance'] = predictive_covariance
            self._pymc_update_model_with_deterministics(pymc_model)
        self._pymc_model = pymc_model
        return self._pymc_model

    def _pymc_update_model_with_deterministics(self, pymc_model):
        for d in self.pymc_deterministics:
            dc = d.copy()
            eval = dc['eval']
            del dc['eval']
            doc = dc['doc']
            del dc['doc']
            name = dc['name']
            del dc['name']
            real_parents = {}
            for p in dc['parents']:
                real_parents[p] = pymc_model[p]
            dc['parents'] = real_parents
            d_obj = pm.Deterministic(eval, doc, name, **dc)
            pymc_model[name] = d_obj

    @property
    def pymc_mcmc(self):
        """
        Return a PyMC sampler. By default we assign the AdaptiveMetropolis algorithm
        as the sampler of choice.
        """
        if self._pymc_mcmc is not None:
            return self._pymc_mcmc
        self._pymc_mcmc = pm.MCMC(self.pymc_model, db=self.pymc_db, **self.pymc_db_opts)
        self._pymc_mcmc.use_step_method(self.pymc_step_method,
                                        self.pymc_model['transformed_hyperparameters'],
                                        **self.pymc_step_method_params)
        return self._pymc_mcmc

    @property
    def log_evidence(self):
        """
        Compute the evidence using the method of Newton and Raftery (1994).

        The method is really bad and, actually, does not work.
        The reasons are explained in this post of Neal Radford:
        https://radfordneal.wordpress.com/2008/08/17/the-harmonic-mean-of-the-likelihood-worst-monte-carlo-method-ever/

        The only reason I implement it here, is because it is the easiest thing
        to do. So, consider it a placeholder for something better.
        """
        return self.get_log_evidence()

    def get_log_evidence(self, last_idx=-1):
        log_like = self.pymc_mcmc.trace('log_likelihood')[:last_idx]
        return self._get_log_evidence(log_like)

    def _get_log_evidence(self, log_like):
        n = log_like.shape[0]
        return np.log(n) - logsumexp(-log_like)

    def get_log_evidence_history(self):
        log_like = self.pymc_mcmc.trace('log_likelihood')[:]
        n = log_like.shape[0]
        res = np.ndarray((n-2,))
        for i in xrange(n-2):
            res[i] = self._get_log_evidence(log_like[:i+1])
        return res
