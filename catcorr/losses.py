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

# losses.py - loss functions for Tensorflow and Keras built on RK

import tensorflow as tf

from catcorr import core

def log_rk_loss_tf(labels, predictions, name=None):
    """Computes -log(RK) as a loss function between prediction probabilities
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
    loss = tf.cast(loss, tf.float32) # Must cast for autodiff on GPU

    return loss


def rk_loss_tf(labels, predictions, name=None):
    """Computes 1-RK as a loss function between prediction probabilities
    and labels.

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
    RK = core.rk_coeff_tf(C)
    loss = 1 - RK
    loss = tf.cast( loss, tf.float32) # Must cast for autodiff on GPU

    return loss


def rk_loss_fn(from_logits=False):
    """Functor that returns a Keras loss function for Gorodkin's R_K,
     optionally adding softmax layer when predictions are given as
     logits.

    Parameters:
      from_logits : If true, adds a softmax layer to predictions for calculating 
                      the soft confusion matrix
    Returns:
      loss_fn : A binary function for Keras that takes two parameters, labels and
                  predictions

    """
    def loss_fn(labels, predictions):
        if from_logits:
            predictions = tf.keras.activations.softmax( predictions )
        return rk_loss_tf( labels, predictions )
            
    return loss_fn


def log_rk_loss_fn(from_logits=False):
    """Functor that returns a Keras loss function for the log of Gorodkin's R_K,
     optionally adding softmax layer when predictions are given as
     logits. Note that R_K should be positive.
    
    Parameters:
      from_logits : If true, adds a softmax layer to predictions for calculating 
                      the soft confusion matrix
    Returns:
      loss_fn : A binary function for Keras that takes two parameters, labels and
                  predictions
    """
    def loss_fn(labels, predictions):
        if from_logits:
            predictions = tf.keras.activations.softmax( predictions )
        return log_rk_loss_tf( labels, predictions )
            
    return loss_fn


class RkLoss():
    """Class suitable for use as a callable passed to Keras optimizer as loss"""
    
    def __init__(self, from_logits=False):
        self.from_logits = from_logits

        def __call__(self, labels, predictions):
            if self.from_logits:
                predictions = tf.keras.activations.softmax( predictions )
            return rk_loss_tf( labels, predictions )

        def get_config(self):
            """config dictionary for serialization protobuffer"""
            return {"from_logits": self.from_logits}

    
