# core.py - A suite of routines to tabulate soft confusion matrices
# and correlations for NumPy and Tensorflow

"""
    For background details, refer to http://rk.kvl.dk and

      Gorodkin, J. (2004) Comparing two K-category assignments by a 
         K-category correlation coefficient. J. Comp. Biol. and Chem.,
         28:367--374. https://doi.org/10.1016/j.compbiolchem.2004.09.006
"""

import tensorflow as tf
import numpy as np



def soft_confusion_matrix_tf(labels, predictions,
                             name=None):
    """Computes the (soft) confusion matrix from prediction probabilities
    and labels. 

    The row indices represent the real (ground truth) labels, while
    the column indices represent the predicted labels.

    Prediction probabilities are accumulated in the confusion matrix rows.

    Let N be batch size and L be the number of class labels.

    Parameters:
      labels      : 1-D (rank 1) tensor (length N) of ground-truth labels
                      (indices) in {0,...,L-1} or 2-D (rank 2) NxL tensor
                      of one-hot encoded labels 
      predictions : 2-D (rank 2) tensor (size NxL) probabilities for each
                      class in [0,1]
    Returns:
      C : LxL matrix of the same type as predictions
    """

    # TODO: Verify labels and predictions dimensions match
    label_shape = labels.get_shape().as_list()
    if len(label_shape) == 2 and label_shape[1]!=1: # must be a better way!
        # Convert one-hot to raw indices
        labels = tf.argmax(labels, axis=1) # NB: produces int64 by default
    else: 
        # Ensure we're not getting a float (i.e., from Keras [but why??]) 
        labels = tf.cast(labels, tf.int32)
        labels = tf.squeeze(labels) # Eliminate the extra dimension 
    
    # Ensure we have a complete matrix with the correct number of rows
    num_classes = tf.shape(predictions)[1]

    C = tf.math.unsorted_segment_sum(predictions, labels, num_classes,
                                     name or 'confusion_matrix')

    return C

def _rk_terms_tf(C):
    """
    Calculate the numerator and denominator of Gorodkin's R_K Correlation
    Coefficient from an KxK confusion matrix in TensorFlow.

    Preconditions The type of C (if integer) should be large enough to
      prevent overflow when calculating N*N, where N is the total
      number of elements represented by the confusion matrix (i.e.,
      the reduced sum of C).
    

    Parameters
      C : a KxK confusion matrix Tensor (integer or floating point values)

    Returns
      numerator    : a scalar Tensor
      denominator  : a scalar Tensor
      denominators : a list of two Tensors containing the two raw denominator 
                       terms (before multiplying and the square root)
    """
    # TODO: throw error if confusion matrix is not square

    # Downstream, we will cast to floating point type just in case we were
    # given integer types that normally live in a confusion matrix
    float_type = tf.float64
    
    N = tf.math.reduce_sum(C)    # Total number of observations
    trace_C = tf.linalg.trace(C) # Sum of diagonal (correct predictions)

    # Refer to Eq. (8) in Gorodkin (2004)
    numerator =  N*trace_C - tf.math.reduce_sum(tf.linalg.matmul(C,C))

    # Calculate true and predicted marginals
    row_sum   = tf.reduce_sum(C, axis=1, keepdims=False)
    col_sum   = tf.reduce_sum(C, axis=0, keepdims=False)

    Nsq = N*N
    denominator_1 = Nsq - tf.tensordot(row_sum, row_sum, axes=1)
    denominator_2 = Nsq - tf.tensordot(col_sum, col_sum, axes=1)

    numerator     = tf.dtypes.cast(numerator, float_type)
    denominator_1 = tf.dtypes.cast(denominator_1, float_type)
    denominator_2 = tf.dtypes.cast(denominator_2, float_type)
    
    # Terms inside the denominator must not be negative; if they are
    # it's likely due to limited precision (we hope)
    zero = tf.dtypes.cast(0.0, float_type)
    denominator_1 = tf.math.maximum(zero, denominator_1)
    denominator_2 = tf.math.maximum(zero, denominator_2)

    # Multiply before taking the square root. Although that might
    # limit precision, it should(?) be slightly more efficient
    denominator = tf.math.sqrt( denominator_1 * denominator_2 )

    denominators = [denominator_1, denominator_2]
    
    return numerator, denominator, denominators

def log_rk_coeff_tf(C):
    """
    Calculate Gorodkin's R_K Correlation Coefficient from an KxK
    confusion matrix in TensorFlow.

    Preconditions:
      1) Rk > 0
      2) The type of C (if integer) should be large enough to
      prevent overflow when calculating N*N, where N is the total
      number of elements represented by the confusion matrix (i.e.,
      the reduced sum of C).
    

    Parameters
      C : a KxK confusion matrix Tensor (integer or floating point values)

    Returns
      log_rk : a scalar Tensor
    """
    numerator, _, denominators =  _rk_terms_tf(C)

    log_top = tf.math.log(numerator)
    log_bot = (tf.math.log(denominators[0]) + tf.math.log(denominators[1]))/2

    log_rk = log_top - log_bot

    return log_rk

    
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

    numerator, denominator, _ = _rk_terms_tf(C)

    # NB: The limit of the original Matthews correlation coefficient
    # (MCC) is zero as each of the terms in that denominator's sums
    # approach zero from the right; cf. p. 415 of Baldi et al. (2000),
    # Bioinformatics Review 16(5), 412--424. Following a similar (but
    # not yet theoretically proven) approach here, we avoid dividing
    # by zero and give a quotient of zero.

    rk = tf.math.divide_no_nan( numerator, denominator )

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

    # Calculate true and predicted marginals
    row_sum   = np.sum(C, axis=1, keepdims=False)
    col_sum   = np.sum(C, axis=0, keepdims=False)

    Nsq = N*N
    denominator_1 = Nsq - np.dot(row_sum, row_sum)
    denominator_2 = Nsq - np.dot(col_sum, col_sum)

    
    # Terms inside the denominator must not be negative; if they are
    # it's likely due to limited precision (we hope)
    denominator_1 = np.maximum(0.0, denominator_1)
    denominator_2 = np.maximum(0.0, denominator_2)

    # NB: The limit of the original Matthews correlation coefficient
    # (MCC) is zero as each of the terms in that denominator's sums
    # approach zero from the right; cf. p. 415 of Baldi et al. (2000),
    # Bioinformatics Review 16(5), 412--424. Following a similar (but
    # not yet theoretically proven) approach here, we avoid dividing
    # by zero and give a quotient of zero.

    if (denominator_1==0 or denominator_2 == 0):
        rk = 0
    else:
        rk = numerator / np.sqrt( denominator_1 * denominator_2 )

    return rk
