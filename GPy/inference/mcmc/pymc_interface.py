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

    def __init__(self, pymc_db='ram', pymc_db_opts={}):
        self._pymc_db = pymc_db
        self._pymc_db_opts = pymc_db_opts

    @property
    def pymc_model(self):
        """
        Take the current model and turn it into a PyMC model.
        """
        if self._pymc_model is not None:
            return self._pymc_model
        @pm.stochastic(dtype=np.ndarray)
        def transformed_hyperparameters(value=self.optimizer_array, model=self):
            return -model._objective(value)
        @pm.deterministic(dtype=np.ndarray)
        def hyperparameters(transformed_hyperparameters=transformed_hyperparameters, model=self):
            theta = np.ndarray(transformed_hyperparameters.shape)
            model._inverse_hyperparameter_transform(transformed_hyperparameters, theta)
            return theta
        model = {'transformed_hyperparameters': transformed_hyperparameters}
        model['hyperparameters'] = hyperparameters
        self._pymc_model = model
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
        self._pymc_mcmc.use_step_method(pm.AdaptiveMetropolis,
                                        self.pymc_model['transformed_hyperparameters'])
        return self._pymc_mcmc
