# losses.py - loss functions built on Rk

import tensorflow as tf

from catcorr import core

def log_rk_loss_tf(labels, predictions, name=None):
    """Computes -log(Rk) as a loss function between prediction probabilities
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
      loss : Scalar tensor in (Inf,0]
    """

    C = core.soft_confusion_matrix_tf(labels, predictions)
    log_rk = core.log_rk_coeff_tf(C)
    loss = - log_rk
    loss = tf.cast( loss, tf.float32) # Must cast for autodiff on GPU

    return loss

def rk_loss_tf(labels, predictions, name=None):
    """Computes 1-Rk as a loss function between prediction probabilities
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
      loss : Scalar tensor in [0,2]
    """

    C = core.soft_confusion_matrix_tf(labels, predictions)
    Rk = core.rk_coeff_tf(C)
    loss = 1 - Rk
    loss = tf.cast( loss, tf.float32) # Must cast for autodiff on GPU

    return loss

def rk_loss_fn(from_logits=False):

    def loss_fn(labels, predictions):
        if from_logits:
            predictions = tf.keras.activations.softmax( predictions )
        return rk_loss_tf( labels, predictions )
            
    return loss_fn

def log_rk_loss_fn(from_logits=False):

    def loss_fn(labels, predictions):
        if from_logits:
            predictions = tf.keras.activations.softmax( predictions )
        return log_rk_loss_tf( labels, predictions )
            
    return loss_fn


class RkLoss():
    """Class suitable for use as a callable passed to as Keras loss"""
    
    def __init__(self, from_logits=False):
        self.from_logits = from_logits

        def __call__(self, labels, predictions):
            if self.from_logits:
                predictions = tf.keras.activations.softmax( predictions )
            return rk_loss_tf( labels, predictions )

        def get_config(self):
            """config dictionary for serialization protobuffer"""
            return {"from_logits": self.from_logits}

    
