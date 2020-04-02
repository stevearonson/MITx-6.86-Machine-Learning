"""Mixture model using EM"""
from typing import Tuple
import numpy as np
from common import GaussianMixture



def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data
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
    Px = np.zeros((K, n))
    for ix in range(K):
        Px[ix,:] = mixture.p[ix] * np.exp(-np.linalg.norm(X - mixture.mu[ix], axis=1)**2/(2*mixture.var[ix]))/(2*np.pi*mixture.var[ix])**(d/2)
        
    cost = np.sum(np.log(np.sum(Px, axis=0)))
    post = Px / np.sum(Px, axis=0)
    
    return post.T, cost


def mstep(X: np.ndarray, post: np.ndarray) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    n, d = X.shape
    K = post.shape[1]
    
    nj = np.sum(post, axis=0)
    pj = nj/n
    
    mu = (post.T @ X)/nj.reshape(K,1)
    
    var = np.zeros(K)
    for ix in range(K):
        var[ix] = np.sum(post[:,ix] * np.linalg.norm(X - mu[ix,:], axis=1)**2)/(d*nj[ix])
    
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
        mixture = mstep(X, post)

    return mixture, post, cost
