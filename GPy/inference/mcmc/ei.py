"""
An implmentation of the expected improvement data acquisition function.

Author:
    Ilias Bilionis

Date:
    5/1/2015

"""


__all__ = ['expected_improvement']



import numpy as np
import scipy.stats as stats


def expected_improvement(predictive_mean, predictive_variance,
                         Y_obs, mode='min',
                         noise=0.):
    """
    Compute the expected improvement at ``Xd``.

    :param predictive_mean: The posterior mean on the design points.
    :param predictive_variance: The posterior variance on the design points.
    :param m_obs:   The observed outputs.
    :param noise:   The variance of the measurement noise on each design point. If
                    ``None``, then we attempt to get this noise from
                    ``model.likelihood.noise``, if possible.
    :returns:       The expected improvement on all design points.
    """
    #m_obs = model.predict(X)[0].flatten()
    m_obs = Y_obs.flatten()
    m_s = predictive_mean
    v_s = predictive_variance
    m_s = m_s.flatten()
    v_s = v_s.flatten() - noise
    s_s = np.sqrt(v_s)
    idx = np.isnan(s_s)
    s_s[np.isnan(s_s)] = 1e-10
    if mode == 'min':
        i_n = np.argmin(m_obs)
        m_n = m_obs[i_n]
        u = (m_n - m_s) / s_s
    elif mode == 'max':
        i_n = np.argmax(m_obs)
        m_n = m_obs[i_n]
        u = (m_s - m_n) / s_s
    else:
        raise NotImplementedError('I do not know what to do with mode %s' %mode)
    ei = s_s * (u * stats.norm.cdf(u) + stats.norm.pdf(u))
    return ei
