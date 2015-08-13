# Written by Ilias Bilionis (ibilion@purdue.edu)
"""
Methods that allow one to interface GPy with PyMC.

We keep it as simple as possible.
"""


import pymc as pm
import numpy as np


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

    def __init__(self, pymc_db='ram', pymc_db_opts={},
                 pymc_step_method=pm.AdaptiveMetropolis,
                 pymc_step_method_params={},
                 X_predict=None, num_predict=100):
        self._pymc_db = pymc_db
        self._pymc_db_opts = pymc_db_opts
        self._pymc_step_method = pymc_step_method
        self._pymc_step_method_params = pymc_step_method_params
        self._X_predict = X_predict
        self._num_predict = num_predict

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
            return obj
        @pm.stochastic(observed=True, trace=False)
        def observation(value=1., model=model, theta=transformed_hyperparameters):
            return -model.objective_function()
        @pm.deterministic(dtype=np.ndarray)
        def hyperparameters(transformed_hyperparameters=transformed_hyperparameters, model=model):
            theta = np.ndarray(transformed_hyperparameters.shape)
            model._inverse_hyperparameter_transform(transformed_hyperparameters, theta)
            return theta
        @pm.deterministic(dtype=float)
        def log_likelihood(model=model, theta=transformed_hyperparameters):
            model.optimizer_array = theta
            return model.log_likelihood()
        @pm.deterministic(dtype=float)
        def log_prior(model=model, theta=transformed_hyperparameters):
            model.optimizer_array = theta
            return model.log_prior()
        pymc_model = {'transformed_hyperparameters': transformed_hyperparameters}
        pymc_model['hyperparameters'] = hyperparameters
        pymc_model['log_likelihood'] = log_likelihood
        pymc_model['log_prior'] = log_prior
        if self.X_predict is not None:
            @pm.deterministic(dtype=np.ndarray, trace=False)
            def full_prediction(model=model, theta=transformed_hyperparameters):
                model.optimizer_array = theta
                return model.predict(model.X_predict, full_cov=True)
            @pm.deterministic(dtype=np.ndarray)
            def predictive_mean(full_prediction=full_prediction):
                return full_prediction[0]
            @pm.deterministic(dtype=np.ndarray)
            def predictive_covariance(full_prediction=full_prediction):
                return full_prediction[1]
            @pm.deterministic(dtype=np.ndarray)
            def posterior_samples(mu=predictive_mean, C=predictive_covariance,
                                  model=model, theta=transformed_hyperparameters):
                model.optimizer_array = theta
                return np.random.multivariate_normal(mu.flatten(), C, model.num_predict)
            pymc_model['full_prediction'] = full_prediction
            pymc_model['predictive_mean'] = predictive_mean
            pymc_model['predictive_variance'] = predictive_covariance
            pymc_model['posterior_samples'] = posterior_samples
        self._pymc_model = pymc_model
        return self._pymc_model

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
