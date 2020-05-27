# metrics.py - Folderoll (classes, tensors, ops) for metrics supporting Rk

import tensorflow as tf
import tensorflow.keras

from catcorr import catcorr

""" For background details, refer to http://rk.kvl.dk and

      Gorodkin, J. (2004) Comparing two K-category assignments by a 
         K-category correlation coefficient. J. Comp. Biol. and Chem.,
         28:367--374. https://doi.org/10.1016/j.compbiolchem.2004.09.006
"""

# rk_coeff documentation modeled after tf.compat.v1.metrics.accuracy
# <https://www.tensorflow.org/api_docs/python/tf/compat/v1/metrics/accuracy>
# Used under the terms of a Creative Commons Attribution 4.0 License
# <https://creativecommons.org/licenses/by/4.0/>

def rk_coeff(labels, predictions, num_classes,
             streaming=True,
             name=None):
    """Calculates Gorodkin's Rk correlation coefficient.

    The `rk_coeff` function creates a local variable, `table` to store
    the confusion matrix used to compute the Rk coefficient. That value
    is ultimately returned as `coeff`: an idempotent operation.

    For estimation of the metric over a stream of data (indicated by
    `streaming`), the function creates an `update_op` operation that
    updates these variables and returns the `coeff`. The `update_op`
    accumulates values in the confusion matrix.

    Let N be batch size and L be the number of class labels.

    Parameters:
      labels      : 1-D (rank 1) tensor (length N) of ground-truth labels
                      (indices) in {0,...,L-1} or 2-D (rank 2) NxL tensor
                      of one-hot encoded labels 
      predictions : 2-D (rank 2) tensor (size NxL) probabilities for each
                      class in [0,1]
      num_classes : An scalar integer value indicating the number of class
                      labels (length of one side of the confusion matrix
                      table)
    Returns:
      coefficient : A `Tensor` representing the Rk coefficient of the `table`.
      update_op :   An operation that increments the `table` variable
                      appropriately and whose value matches `coefficient`.

    """

    #var_collections = [ tf.compat.v1.GraphKeys.LOCAL_VARIABLES,
    #                    tf.compat.v1.GraphKeys.METRIC_VARIABLES ]
    
    batch_table = catcorr.soft_confusion_matrix_tf(
        labels, predictions, name='rk_coeff/batch_table' )

    # After futzing for hours, I could not get a more elegant solution
    # inferring the dimensions of batch_table to work with v1-oriented
    # TF code, (it is a variale initialization issue) which
    # necessitates the num_classes argument to hard code the
    # dimensions

    # This line in and of itself works, but fails when passed to tf.Variable
    #zero_table = tf.zeros_like(batch_table)
    # So we fall back to the following:
    zero_table = tf.zeros([num_classes,num_classes],dtype=batch_table.dtype)

    #table = tf.compat.v1.Variable( zero_table, dtype=batch_table.dtype),
    #                               trainable=False,
    #                               name='rk_coeff/table',
    #                               collections=var_collections,
    #                               aggregation=tf.VariableAggregation.SUM )
    table = tf.Variable( zero_table,
                         trainable=False,
                         validate_shape=True,
                         name='rk_coeff/table',
                         dtype=batch_table.dtype,
                         aggregation=tf.VariableAggregation.SUM )

    if streaming: # Accumulate confusion matrix
        update_table_op = tf.compat.v1.assign_add(table, batch_table) # +=
    else: # Reset confusion matrix
        update_table_op = tf.compat.v1.assign(table, batch_table)     # := 

    coeff = tf.identity( catcorr.rk_coeff_tf(table), name='rk_coeff/value')
    update_op = tf.identity( catcorr.rk_coeff_tf(update_table_op),
                             name='rk_coeff/update_op')
    return coeff, update_op


class RkMetric(tf.keras.metrics.Metric):
    """
    RkMetric: Stores the confusion matrix as table, which can
    accumulate values with the update_state method, produce the current
    Rk coefficient with the result method, and reset the confusion matrix
    accumulator with the reset_states method.
    """

    def __init__(self,
                 num_classes,
                 name: str ='Rk_Metric',
                 dtype = None,
                 **kwargs ):
        super().__init__(name=name, dtype=dtype)
        self.num_classes = num_classes
        self.table = self.add_weight( "Confusion_matrix",
                                      shape = [self.num_classes,
                                               self.num_classes],
                                      initializer="zeros",
                                      dtype=self.dtype )

        
    def update_state( self, labels, predictions ):
        """ 
        Calculate the batch confusion matrix and add it to the accumulator 
        table.

        Parameters:
           labels  : 1-D (rank 1) tensor of ground-truth labels
                      (indices) or 2-D (rank 2) tensor of one-hot encoded
                      labels (batch dim first)
           predictions: dict with key 'probabilities' having a 2-D tensor
                      (batch dim first) of probabilities for each class 
                      in [0,1]
        """
        batch_table = catcorr.soft_confusion_matrix(labels,
                                              predictions['probabilities'])
        batch_table = tf.cast(batch_table, self.dtype)
        self.table.assign_add(batch_table)

        
    def result(self):
        """Give the Rk coefficient"""
        return catcorr.rk_coeff_tf(self.table)


    def reset_states(self):
        """Reset confusion matrix to all zeros"""
        self.table.assign(tf.zeros((self.num_classes, self.num_classes), self.dtype))
