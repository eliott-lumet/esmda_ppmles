# -*- coding: utf-8 -*-
"""
Auxiliary functions.

@author: lumet
"""
# =============================================================================
# Imports
# =============================================================================
import numpy as np
import openturns as ot

from scipy.stats import truncnorm

# =============================================================================
# Functions
# =============================================================================
def min_max_normalize(x, xmin, xmax, s=1):
    """
    Rescaling a set of data from [xmin, xmax] to [0., 1.].
    """
    return s * (x - xmin)/(xmax - xmin) 

def log_transform_cut(y, threshold):
    """
    Apply a log transformation with threshold to the all the value of a 
    dataset that are > 0.
    """
    return np.log(np.where(y>=0, y, 0) + threshold)

def is_diagonal_matrix(M):
    """
    Checks if M is a diagonal matrix
    """
    return np.all(M == np.diag(np.diagonal(M)))

def halton_multivariate_truncated_normal(mean, cov, lower_bounds, upper_bounds,
                                         n_samples):
    """
    Draw samples from independent truncated normal distributions using Halton's
    sequence.

    Parameters
    ----------
    mean : np.array, (d,)
        Array of the mean of the d variables that will be sampled.
    cov : np.array, (d, d)
        Covariance matrice of the variables. It has to be diagonal as the 
        variables are assumed to be independent. Each diagonal value corresponds
        to the square value of one variable standard deviation.
    lower_bounds : np.array, (d,)
        Array of the lower bounds applied to truncate each distribution. Use 
        -np.inf to avoid truncation.
    upper_bounds : np.array, (d,)
        Array of the upper bounds applied to truncate each distribution. Use
        np.inf to avoid truncation.
    n_samples : int
        Number of samples that will be drawn.

    Returns
    -------
    samples : np.array, (d, n_samples)
        Array of the n_samples of d independant variables.

    """    
    # Parameters checking:
    d = len(mean)
    
    if not is_diagonal_matrix(cov):  # Check if the covariance matrix is diagonal
        raise Exception('Error: the halton_multivariate_truncated_normal sampling function only handles diagonal cov matrices')
        
    elif (np.shape(cov)[0] != d) or (np.shape(cov)[1] != d):
        raise Exception(f"Error: the covariance matrice shape does not match the mean vector shape: {np.shape(cov)} != ({d},{d}).")

    elif (np.shape(lower_bounds) != np.shape(mean)):
        raise Exception(f"Error: the shape of the upper_bounds vector does not match the mean vector shape: {np.shape(lower_bounds)} != {np.shape(mean)}.")

    elif (np.shape(upper_bounds) != np.shape(mean)):
        raise Exception(f"Error: the shape of the upper_bounds vector does not match the mean vector shape: {np.shape(upper_bounds)} != {np.shape(mean)}.")

    elif np.any(np.where(lower_bounds > upper_bounds, True, False)):
        for k in range(d):  # Finds the index of the error
            if lower_bounds[k] > upper_bounds[k]:
                raise Exception(f"Error: one lower_bound ({lower_bounds[k]:.2f}) is larger than the corresponding upper bound ({upper_bounds[k]}).")

    else:        
        # Generate the Halton sequence
        halton_seq = ot.HaltonSequence(d)
        halton_samples = np.array(halton_seq.generate(n_samples))
        samples = np.zeros(np.shape(halton_samples))
        for k in range(len(mean)): 
            if mean[k] - 2*np.sqrt(cov[k,k]) < lower_bounds[k]:
                print(f'WARNING: the [mean - 2*sigma, mean + 2*sigma] interval exceeds the lower bound of the truncated normal distribution for the variable #{k+1}.')
                print(f'{mean[k] - 2*np.sqrt(cov[k,k])} < {lower_bounds[k]}\nIt could induce significant differences between the normal law and the truncated one.') 
            if mean[k] + 2*np.sqrt(cov[k,k]) > upper_bounds[k]:
                print(f'WARNING: the [mean - 2*sigma, mean + 2*sigma] interval exceeds the upper bound of the truncated normal distribution for the variable #{k+1}.')
                print(f'{mean[k] + 2*np.sqrt(cov[k,k])} > {upper_bounds[k]}\nIt could induce significant differences between the normal law and the truncated one.') 

            a = (lower_bounds[k] - mean[k]) / np.sqrt(cov[k,k])
            b = (upper_bounds[k] - mean[k]) / np.sqrt(cov[k,k])

            samples[:,k] = truncnorm.ppf(halton_samples[:,k], a, b, loc=mean[k], 
                                         scale=np.sqrt(cov[k,k]))
    
    return samples