import os

import numpy as np
import tensorflow as tf

# Make directory if does not already exist
def makedirs(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

# Epsilon for stability                                                                                                   
def log(x):
    return tf.log(x + 1e-8)


## Samplers
# Draw samples from joint distribution
def sample_pxy(images, labels, N_subset, mb):
    """
    draw x,y pairs
    """
    sample_idx_j = np.random.choice(N_subset, mb, replace=False)
    _xj = images[sample_idx_j, :]
    _yj = labels[sample_idx_j, :]
    return _xj, _yj

# Draw samples from x marginal distribution
def sample_px(images, N_subset, mb):
    """
    draw images only
    """
    sample_idx_mx = np.random.choice(N_subset, mb, replace=False)
    _xm = images[sample_idx_mx, :]
    return _xm

# Draw samples from y marginal distribution
def sample_py(labels, N_subset, mb):
    """
    draw labels only
    """
    sample_idx_my = np.random.choice(N_subset, mb, replace=False)
    _ym = labels[sample_idx_my, :]
    return _ym


## Performance evaluators
# Mutual information
def evaluate_mi(sess, f_j_mean, x, y, eval_images, eval_labels, N_samples, training=None, mb = 50):

    """
    array of average MI for each minibatch. if N_samples = 100
    and mb = 25, then mi_mb is an array of 4. each element in 
    mi_mb holds the average nats for a particular minibatch.
    """
    mi_mb = np.zeros(N_samples // mb)  

    # for each minibatch in eval_dataset
    for j in range(len(mi_mb)):
        
        # fetch a minibatch
        _xj = eval_images[j*mb:(j+1)*mb]
        _yj = eval_labels[j*mb:(j+1)*mb]

        # Test
        if training is None:
            feed_dict = {x: _xj, y: _yj}
        else:
            feed_dict = {x: _xj, y: _yj, training: False}

        """
        f_j_mean represents nats. the more likely (x, y) co-occured,
        the higher the nats.
        """
        mi_mb[j] = sess.run(f_j_mean, feed_dict=feed_dict)

    mi = np.mean(mi_mb)
    return mi 

# Accuracy (joint vs marginals)
def evaluate_acc(sess, accuracy, x, y, eval_images, eval_labels, N_samples, joint, training=None, mb = 50):
    acc_mb = np.zeros(N_samples // mb)
    if not joint:
        eval_labels = np.random.permutation(eval_labels)

    for j in range(len(acc_mb)):
    # Sample data
        _xj = eval_images[j*mb:(j+1)*mb]
        _yj = eval_labels[j*mb:(j+1)*mb]

        # Test              
        if training is None:
            feed_dict = {x: _xj, y: _yj}
        else:
            feed_dict = {x: _xj, y: _yj, training: False}
        acc_mb[j] = sess.run(accuracy, feed_dict=feed_dict)

    acc = np.mean(acc_mb)
    return acc
