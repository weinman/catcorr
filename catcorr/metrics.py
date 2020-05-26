# RkMetric.py - Class to track Rk value using the Estimator Metric functionality

import tensorflow as tf
import tensorflow.keras

import catcorr

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
