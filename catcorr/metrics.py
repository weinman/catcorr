# CatCorr
# Copyright (c) 2020 Jerod Weinman and Nathan Gifford
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# metrics.py - Folderoll (classes, tensors, ops) for metrics supporting Rk

import tensorflow as tf

from catcorr import core


# rk_coeff documentation modeled after tf.compat.v1.metrics.accuracy
# <https://www.tensorflow.org/api_docs/python/tf/compat/v1/metrics/accuracy>
# Used under the terms of a Creative Commons Attribution 4.0 License
# <https://creativecommons.org/licenses/by/4.0/>

def rk_coeff(labels, predictions, num_classes,
             streaming=True,
             name=None):
    """Tensorflow metric calculating Gorodkin's Rk correlation coefficient.

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
      num_classes : A scalar integer value (or tensor) indicating the number
                      of class labels (length of one side of the confusion
                       matrix table)
    Returns:
      coefficient : A `Tensor` representing the Rk coefficient of the `table`.
      update_op :   An operation that increments the `table` variable
                      appropriately and whose value matches `coefficient`.

    """

    #var_collections = [ tf.compat.v1.GraphKeys.LOCAL_VARIABLES,
    #                    tf.compat.v1.GraphKeys.METRIC_VARIABLES ]
    
    batch_table = core.soft_confusion_matrix_tf(
        labels, predictions, name='rk_coeff/batch_table' )

    # After futzing for hours, I could not get a more elegant solution
    # inferring the dimensions of batch_table to work with v1-oriented
    # TF code, (it is a variable initialization issue) which
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

    coeff = tf.identity( core.rk_coeff_tf(table), name='rk_coeff/value')
    update_op = tf.identity( core.rk_coeff_tf(update_table_op),
                             name='rk_coeff/update_op')
    return coeff, update_op


class RkMetric(tf.keras.metrics.Metric):
    """
    Keras metric for Gorodkin's RK. Stores the confusion matrix as
    table, which can accumulate values with the update_state method,
    produce the current Rk coefficient with the result method, and
    reset the confusion matrix accumulator with the reset_states
    method.
    """
    
    def __init__(self,
                 num_classes,
                 name: str ='Rk',
                 dtype = tf.float32,
                 from_logits: bool = False,
                 **kwargs ):
        """RkMetric

        Parameters
          num_classes : A scalar integer value (or tensor) indicating the 
                          number of class labels (length of one side of the
                          confusion matrix table)
          name        : String description to display with metric value
          dtype       : Data type to use for storing the confusion matrix
          from_logits : Boolean Indicating whether to apply a softmax 
                          operation to the given prediction values before
                          calculating the confusion matrix
        """
        super().__init__(name=name, dtype=dtype)
        self.num_classes = num_classes
        self.from_logits = from_logits
        self.table = self.add_weight( "confusion_matrix",
                                      shape = [self.num_classes,
                                               self.num_classes],
                                      initializer="zeros",
                                      dtype=self.dtype )


    def get_config(self):
        """config dictionary for serialization protobuffer"""
        config = super(RkMetric, self).get_config()
        config.update({"num_classes": self.num_classes,
                       "from_logits": self.from_logits,
        })
        return config

    
    @classmethod
    def from_config(cls, config):
        return cls(**config)

    
    def update_state( self, labels, predictions ):
        """ Calculate the batch confusion matrix and add it to the 
        accumulator table.

        Parameters:
           labels      : 1-D (rank 1) tensor of ground-truth labels
                           (indices) or 2-D (rank 2) tensor of one-hot encoded
                           labels (batch dim first)
           predictions : a 2-D tensor of shape (?, num_classes) with the
                            values (logits or probabilities) for each class 
        """
        if self.from_logits:
            predictions = tf.keras.activations.softmax(predictions)
        batch_table = core.soft_confusion_matrix_tf( labels, predictions )
        batch_table = tf.cast(batch_table, self.dtype)
        self.table.assign_add(batch_table)

        
    def result(self):
        """Give the Rk coefficient"""
        return core.rk_coeff_tf(self.table)


    def reset_states(self):
        """Reset confusion matrix to all zeros"""
        self.table.assign( tf.zeros( (self.num_classes, self.num_classes),
                                     self.dtype))
