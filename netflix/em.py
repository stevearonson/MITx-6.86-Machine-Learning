"""Mixture model for matrix completion"""
from typing import Tuple
import numpy as np
from scipy.special import logsumexp
from common import GaussianMixture


def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment

    """
    n, d = X.shape
    K = mixture.mu.shape[0]
    
    '''
    calculate N(X, mu, var) for each cluster
    '''
    # identify non zero elements of X
    valid_X = X != 0
    validd = np.sum(valid_X, axis=1)
    
    # compute Gaussian probabilites of each data point for each cluster
    Px_log = np.zeros((K, n))
    for k in range(K):
        Px_log[k,:] = np.log(mixture.p[k])
        Px_log[k,:] -= np.sum(((X - mixture.mu[k]) * valid_X) **2, axis=1)/(2*mixture.var[k])
        Px_log[k,:] -= (validd/2) * np.log(2*np.pi*mixture.var[k])
        
    cost = np.sum(logsumexp(Px_log, axis=0))
    post_log = Px_log - logsumexp(Px_log, axis=0)
    
    return np.exp(post_log.T), cost



def mstep(X: np.ndarray, post: np.ndarray, mixture: GaussianMixture,
          min_variance: float = .25) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        post: (n, K) array holding the soft counts
            for all components for all examples
        mixture: the current gaussian mixture
        min_variance: the minimum variance for each gaussian

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    n, d = X.shape
    K = post.shape[1]
    
    # identify non zero elements of X
    valid_X = X != 0
    validd = np.sum(valid_X, axis=1)

    nj = np.sum(post, axis=0)
    pj = nj/n
    
    mu = (post.T @ X) / (post.T @ valid_X)
    # ignore new mu calculations when sample support is less than 1
    keep_mu = post.T @ valid_X < 1
    if np.sum(keep_mu) > 0:
        mu[keep_mu] = mixture.mu[keep_mu]
    
    var = np.zeros(K)
    for k in range(K):
        var[k] = np.sum(post[:,k] * np.sum(((X - mu[k]) * valid_X) **2, axis=1)) / np.sum(validd * post[:,k])
    # use a var floor of 0.25
    var = np.maximum(var, min_variance)
    
    return GaussianMixture(mu, var, pj)


def run(X: np.ndarray, mixture: GaussianMixture,
        post: np.ndarray) -> Tuple[GaussianMixture, np.ndarray, float]:
    """Runs the mixture model

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the current assignment
    """
    prev_cost = None
    cost = None
    while (prev_cost is None or np.abs(prev_cost - cost) >= np.abs(cost) * 1e-6):
        prev_cost = cost
        post, cost = estep(X, mixture)
        mixture = mstep(X, post, mixture)

    return mixture, post, cost


def fill_matrix(X: np.ndarray, mixture: GaussianMixture) -> np.ndarray:
    """Fills an incomplete matrix according to a mixture model

    Args:
        X: (n, d) array of incomplete data (incomplete entries =0)
        mixture: a mixture of gaussians

    Returns
        np.ndarray: a (n, d) array with completed data
    """
    n, d = X.shape
    K = mixture.mu.shape[0]
    
    '''
    calculate N(X, mu, var) for each cluster
    '''
    # identify non zero elements of X
    valid_X = X != 0
    validd = np.sum(valid_X, axis=1)
    
    # compute Gaussian probabilites of each data point for each cluster
    Px_log = np.zeros((K, n))
    for k in range(K):
        Px_log[k,:] = np.log(mixture.p[k])
        Px_log[k,:] -= np.sum(((X - mixture.mu[k]) * valid_X) **2, axis=1)/(2*mixture.var[k])
        Px_log[k,:] -= (validd/2) * np.log(2*np.pi*mixture.var[k])
        
    post = np.exp(Px_log - logsumexp(Px_log, axis=0)).T
    
    X_hat = post @ mixture.mu
    
    X_fill = X.copy()
    X_fill[~valid_X] = X_hat[~valid_X]
    
    return X_fill
