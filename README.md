# CatCorr: Correlation Coefficient for Multi-Class Prediction

**Authors: Jerod Weinman and Nathan Gifford**

Numpy, Tensorflow, and (tf)Keras routines implementing Gorodkin's
K-category correlation coefficient (R_K). The formulation is a
generalization of the binary (two-class) quantity known in statistics
as Pearson's φ coefficient and bioinformatics at the Matthew's
correlation coefficient (MCC).

For theoretical details, see the original paper introducing the coefficient:

> Gorodkin, Jan (2004). "Comparing two K-category assignments by a K-category correlation coefficient". Computational Biology and Chemistry. 28 (5): 367–374. doi:[10.1016/j.compbiolchem.2004.09.006](https://doi.org/10.1016/j.compbiolchem.2004.09.006). PMID 15556477

Readers may also be interested in Gorodkin's
[RK page](http://rk.kvl.dk/introduction/index.html), where `perl` and
`awk` implementations are available.

The implementations work with Tensorflow 1.15 or 2.x.

## Overview

For real-valued terms, Pearson's correlation coefficient ρ is given as 

ρ = cov(X,Y) / √(cov(X,X) * cov(Y,Y))

The derivation of R_K begins with the analog form—see Equation (2) in
Gorodkin (2004).

For discrete classification, Pearson's two-class
[phi coefficient](https://en.wikipedia.org/wiki/Phi_coefficient) may
be calculated as

φ = (TP * TN - FP * FN) / √( (TP+FN) * (TP+FP) * (TN+FP) * (TN+FN) ),

where TP is the number of true positives, FN the number of false
negatives, and so on. Several alternate formulations exist.

The generalization to the multi-class case was derived by Gorodkin
(2004), using the prediction confusion matrix C for the contingency
table, as

R = (N * Tr(C) - C ⨯ C) / √( (N² - P⋅P) * (N² - R⋅R) ),

where N is the total sample size (the sum of all the entries in C),
Tr(C) is the trace of the matrix C (the sum along the diagonal, giving
the tally of correct predictions), P is the vector representing the
row sum of C (the marginal histogram of true class labels), R is the
vector representing the column sum of C (the marginal historgram of
predicted class labels), * the general scalar product, ⨯ a matrix
product, and ⋅ the vector dot product.

This represents the multi-class generalization of the Matthew's
correlation coefficient, or Pearson's φ (phi).

Like Pearson's r and φ (phi), R ∈ [-1,1].

The implementation allows a "soft" confusion matrix.

## Usage

### Tensorflow

#### Loss

To invoke as a loss function, put the root `catcorr` directory in
your `PYTHONPATH`. Then one may write something like the following

```python
import catcorr.losses

# Set up code, etc. ...

# For batch size N, and a model with K classes
labels = ... # an N-length vector or NxK (i.e., one-hot) ground truths
probabilities = ... # NxK (i.e., softmax) predicted outputs 
loss = catcorr.losses.rk_loss_tf(labels,probabilities)
```

Alternatively, if one has done some pretraining (to ensure the value
is positive), one may use `catcorr.losses.log_rk_loss_tf` as a more
numerically stable version of the coefficient's logarithm, expanding
the loss function's dynamic range.

Note that if your model produces logits, you must add a
`tf.nn.softmax` layer to the outputs before passing as the second
input to `rk_loss_tf`.

#### Metric

To track RK (i.e., for text output or a Tensorboard summary), a metric
op is provided. The `streaming` argument is used to indicate that the
value should be calculated "from scratch" for each batch
evaluation(`streaming=False`), or the confusion matrix accumulated
over all batches (`streaming=True`).

```python
import tensorflow as tf
import catcorr.metrics

# Set up code, etc. ...

# For batch size N, and a model with K classes
labels = ... # an N-length vector or NxK (i.e., one-hot) ground truths
probabilities = ... # NxK (i.e., softmax) predicted outputs 

rk_coeff = catcorr.metrics.rk_coeff(
             labels,
             probabilities,
             num_classes,
             streaming=(mode != tf.estimator.ModeKeys.TRAIN) )
             
metrics = {'Rk': rk_coeff }
tf.summary.scalar('Rk, rk_coeff[1] )

tf.estimator.EstimatorSpec(
  ...
  eval_metric_ops=metrics
)
```

### Tensorflow Keras

#### Loss

To train in Tensorflow's Keras with RK as a loss function, use
`catcorr.losses.rk_loss_fn`. To track RK with a metric, use
`catcorr.metrics.RkMetric`. For example:

```python
import tensorflow as tf
import catcorr.losses

# Set up a model
model = tf.keras.Sequential(...)

# Set up metrics (NUM_CLASSES=K)
metrics=[  tf.keras.metrics.SparseTopKCategoricalAccuracy(k=1,name='acc'),
           catcorr.metrics.RkMetric(num_classes=NUM_CLASSES,
                                    from_logits=True) ]
# Compile 
model.compile(
  optimizer=tf.keras.Optimizer.SGD(learning_rate=0.1, momentum=0.9)
  loss=catcorr.losses.rk_loss_fn(from_logits=True),
  metrics=metrics)
  
# ... e.g., model.fit(...)
```

### NumPy

To use in NumPy, one must already have the square confusion matrix,
`conf_mat`, then call `catcorr.core.rk_coeff_np`, as in

```python
import sklearn.metrics
import catcorr.core


conf_mat = sklearn.metrics.confusion_matrix(labels,pred)
# Note that sklearn requires hard predictions, but the underlying catcorr code
# for Rk allows for soft confusion matrix

rk = catcorr.core.rk_coeff_np(conf_mat)
```

### Unit Tests

A few simple unit tests are provided, though they are not entirely
comprehensive they provide a quick sanity check. To invoke them,
invoke the following from within the `catcorr` repository directory:

```
python ./test.py
```
