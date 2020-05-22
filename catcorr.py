# catcorr.py - A suite of routines to tabulate soft confusion matrices
# and correlations for NumPy and Tensorflow

"""
    For background details, refer to http://rk.kvl.dk and

      Gorodkin, J. (2004) Comparing two K-category assignments by a 
         K-category correlation coefficient. J. Comp. Biol. and Chem.,
         28:367--374. https://doi.org/10.1016/j.compbiolchem.2004.09.006
"""

import tensorflow as tf
import numpy as np
import math


def rk_coeff_tf(C):

    """
    Calculate Gorodkin's R_K Correlation Coefficient from an KxK
    confusion matrix in TensorFlow.

    Preconditions The type of C (if integer) should be large enough to
      prevent overflow when calculating N*N, where N is the total
      number of elements represented by the confusion matrix (i.e.,
      the reduced sum of C).
    

    Parameters
      C : a KxK confusion matrix Tensor (integer or floating point values)

    Returns
      rk : a scalar Tensor

    """

    # TODO: throw error if confusion matrix is not square

    # Downstream, we will cast to floating point type just in case we were
    # given integer types that normally live in a confusion matrix
    float_type = tf.float64
    
    N = tf.math.reduce_sum(C)    # Total number of observations
    trace_C = tf.linalg.trace(C) # Sum of diagonal (correct predictions)

    # Refer to Eq. (8) in Gorodkin (2004)
    numerator =  N*trace_C - tf.math.reduce_sum(tf.linalg.matmul(C,C))

    # Calculate true and predicated marginals
    row_sum   = tf.reduce_sum(C, axis=1, keepdims=True) # 1xK
    row_sum_t = tf.transpose(row_sum)                   # Kx1
    col_sum   = tf.reduce_sum(C, axis=0, keepdims=True) # Kx1
    col_sum_t = tf.transpose(col_sum)                   # 1xK

    Nsq = N*N
    denominator_1 = Nsq - tf.reduce_sum(tf.linalg.matmul(row_sum_t, row_sum))
    denominator_2 = Nsq - tf.reduce_sum(tf.linalg.matmul(col_sum, col_sum_t))

    numerator     = tf.dtypes.cast(numerator, float_type)
    denominator_1 = tf.dtypes.cast(denominator_1, float_type)
    denominator_2 = tf.dtypes.cast(denominator_2, float_type)
    
    # Terms inside the denominator must not be negative; if they are
    # it's likely due to limited precision
    zero = tf.dtypes.cast(0.0, float_type)
    denominator_1 = tf.math.maximum(zero, denominator_1)
    denominator_2 = tf.math.maximum(zero, denominator_2)

    # Multiply before taking the square root. Although that might
    # limit precision, it should(?) be slightly more efficient
    denominator = tf.math.sqrt( denominator_1 * denominator_2 )


    # NB: The limit of the original Matthews correlation coefficient
    # (MCC) is zero as each of the terms in that denominator's sums
    # approach zero from the right; cf. Baldi et al. (2000),
    # Bioinformatics Review 16(5), 412--424. Following a similar (but
    # not yet theoretically proven) approach here, we avoid dividing
    # by zero as follows.

    # First, check whether the denominator is zero
    is_bottom_zero = tf.equal(denominator, 0)
    # Next, we modify it ONLY if zero by adding a 1 if so
    safe_denominator = denominator + tf.cast(is_bottom_zero, float_type)
    # Now we can safely divide, because there is no zero in the denominator
    safe_rk = numerator / safe_denominator
    # Last, we need to restore a zero where there should in fact be a zero
    # (by the argument above)
    rk = safe_rk * tf.cast(tf.logical_not(is_bottom_zero), float_type)

    return rk



def rk_coeff_np(C):
    """
    Calculate Gorodkin's R_K Correlation Coefficient from an KxK
    confusion matrix in NumPy.

    Parameters
      C : a KxK confusion matrix (NumPy array)

    Returns
      rk : a scalar float
    """
    # TODO: Verify that C is square
    
    # TODO: Do we care whether a native Python
    # float or a NumPy ndarray (scalar) is returned?
    
    N = np.sum(C)         # Total number of observations
    trace_C = np.trace(C) # Sum of diagonal (correct predictions)

    # Refer to Eq. (8) in Gorodkin (2004)
    numerator =  N*trace_C - np.sum(np.matmul(C,C))

    # Calculate true and predicated marginals
    row_sum   = np.sum(C, axis=1, keepdims=True) # 1xK
    row_sum_t = row_sum.transpose()              # Kx1
    col_sum   = np.sum(C, axis=0, keepdims=True) # Kx1
    col_sum_t = col_sum.transpose()              # 1xK

    Nsq = N*N
    denominator_1 = Nsq - np.sum(np.matmul(row_sum_t, row_sum))
    denominator_2 = Nsq - np.sum(np.matmul(col_sum, col_sum_t))

    # Terms inside the denominator must not be negative; if they are
    # it's likely due to limited precision
    denominator_1 = np.maximum(0.0, denominator_1)
    denominator_2 = np.maximum(0.0, denominator_2)

    # NB: The limit of the original Matthews correlation coefficient
    # (MCC) is zero as each of the terms in that denominator's sums
    # approach zero from the right; cf. Baldi et al. (2000),
    # Bioinformatics Review 16(5), 412--424. Following a similar (but
    # not yet theoretically proven) approach here, we avoid dividing
    # by zero as follows.

    if (denominator_1==0 or denominator_2 == 0):
        rk = 0
    else:
        rk = numerator / np.sqrt( denominator_1 * denominator_2 )

    return rk
