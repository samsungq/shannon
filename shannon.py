from argparse import ArgumentParser
import os

import numpy as np
from tqdm import trange

import tensorflow as tf

from aux_functions import makedirs, log, sample_pxy, sample_px, sample_py, evaluate_acc, evaluate_mi
from networks import mlp5
from util import fetch_data


# Parse arguments
parser = ArgumentParser()
parser.add_argument("--dataset", type=str, required=True, default="mnist")
parser.add_argument("--output_dir", type=str, required=False, default="./Outputs/MNIST/Default")
parser.add_argument("--trials", type=int, required=False, default=3)  # n_trials for each k subset (since we randomly subset)
parser.add_argument("--train_iters", type=int, required=False, default=4001)  # n_epochs
parser.add_argument("--validation_rate", type=int, required=False, default=500)
args = parser.parse_args()

# Output directory                                                                                             
output_dir = args.output_dir
makedirs(output_dir)


# Import data
X_train, X_test, y_train, y_test = fetch_data(args.dataset, test_size=.10)
N_train = y_train.shape[0]
N_test = y_test.shape[0]
n_dim = X_train[0].shape[0]
n_classes = y_train[0].shape[0]
mb = 128  # Minibatch size

## Model Inputs                                                                                                           
# Joint distribution inputs                                                                                            
# always x, y pairs
x_j = tf.placeholder(tf.float32, [None, n_dim], name="x_j")
y_j = tf.placeholder(tf.float32, [None, n_classes], name="y_j")

# Marginal distribution inputs                                                                                            
x_m = tf.placeholder(tf.float32, [None, n_dim], name="x_m")
y_m = tf.placeholder(tf.float32, [None, n_classes], name="y_m")


## Learned Likelihood
"""
learn p(x, y) and p(x) & p(y). 

there are two models in memory, referenced by (a) f_j 
and (b) f_m. they share weights/variables but are not
identical, as they accept different placeholders.

the final layer of the model is not not clamped to 
[0,1]. since the model is estimating nats, a real number,
the model needs to emit a real number.
"""
with tf.variable_scope("f_psi") as scope:
    f_j = mlp5(x_j, y_j)  # calculates mutual information between <x,y>
    scope.reuse_variables()
    f_m = mlp5(x_m, y_m)

## Loss - this is similar but not identical to binary cross entropy, as f_j and f_m are not clamped to [0,1]
# to reduce f_loss, we want f_j as large as possible
# to reduce f_loss, we want f_m as small as possible
f_loss = -tf.reduce_mean(log(tf.sigmoid(f_j)) + log(1 - tf.sigmoid(f_m)))  # binary cross entropy

# Optimizer                                                                                                               
train_step = tf.train.AdamOptimizer().minimize(f_loss)

# Accuracy - we dont really care about accuracy
accuracy_j = tf.reduce_mean(tf.cast(tf.greater(f_j, 0), tf.float32))
accuracy_m = tf.reduce_mean(tf.cast(tf.less(f_m, 0), tf.float32))

# Mean f_j over minibatch
f_j_mean = tf.reduce_mean(f_j)  # what is the average nats for f_j

# Summaries                                                                                                               
f_j_sum = tf.summary.histogram("f_j_sum", f_j)
f_m_sum = tf.summary.histogram("f_m_sum", f_m)
f_loss_sum = tf.summary.scalar("f_loss_sum", f_loss)
merged_sum = tf.summary.merge_all()

init_op = tf.global_variables_initializer()

                                          
# Select subset of training data                                                                                          
train_subset = np.array([0.005, 0.01, 0.05, 0.10, 0.25, 0.50, 1.00])
N_train_subset = np.array(N_train*train_subset, np.int32)  # preallocate array; N_train*train_subset = number of <x,y> to extract

# Number of times to run per subset
trials = args.trials

with tf.Session() as sess:

    # Loop through all subsets of the data                                                                                
    for k in trange(len(train_subset)):
        
        # Make directory to hold the train subset proportion
        prop_dir = os.path.join(output_dir,"train_"+str(train_subset[k]))
        makedirs(prop_dir)

        # for each subset, we'll calculate mutual information for "trials" times
        f_mi_name = os.path.join(prop_dir, "mi.txt")
        f_mi = open(f_mi_name, "a")
        """
        output:

        there will be "trials" number of trials for each subset 
        percentage (eg 0.05). 

        for each trial, it will print valid_MI k times, where 
        k = (train_iter / validation_rate). if your train_iter
        is 8 and validation_rate is 2, then for each trial, 
        youll print 4x.

        it is ended with test_mi. for pretty charting, comment
        out validation_mi, as we only care about test_mi.

        """

        # actual training
        for j in trange(trials):
            
            # Pick a subset of the data, k=partition index
            # if train_subset[k] = 0.10, then subset_indicies is a list of indices, where len(susbset_indices) = 10% of N_train
            subset_indices = np.random.choice(N_train, N_train_subset[k], replace=False)

            # now we have a subset for a trial
            images = X_train[subset_indices,:]
            labels = y_train[subset_indices,:]

            # Initialize all variables                                                                                
            # running init_op resets variables https://stackoverflow.com/questions/38947754/re-initialize-variables-in-tensorflow
            sess.run(init_op)

            # Summary writer
            # https://medium.com/@anthony_sarkis/tensorboard-quick-start-in-5-minutes-e3ec69f673af                                                                               
            writer = tf.summary.FileWriter(os.path.join(output_dir,"train_"+str(train_subset[k]), "run_"+str(j)), sess.graph)

            # Training regimen                                                                                                
            for i in range(args.train_iters):
                # Sample data

                """
                draw *mb* samples from a given dataset partition.

                j refers to joint, m refers to marginal.
                """

                _xj, _yj = sample_pxy(images, labels, N_train_subset[k], mb)  # randomly draw <x, y> pairs of mb size
                _xm = sample_px(images, N_train_subset[k], mb)  # randomly draw <x>
                _ym = sample_py(labels, N_train_subset[k], mb)  # randomly draw <y>

                # Train                                                                                                       
                feed_dict = {x_j: _xj, y_j: _yj, x_m: _xm, y_m: _ym}
                _, _sum = sess.run([train_step, merged_sum], feed_dict=feed_dict)
                writer.add_summary(_sum, i)

                # Validate after each epochs
                # if i % args.validation_rate == 0:
                    # valid_mi = evaluate_mi(sess, f_j_mean, x_j, y_j, mnist.validation.images, mnist.validation.labels, N_valid)
                    # f_mi.write("valid mi:" + str(valid_mi) + "\t")

                    # valid_acc_j = evaluate_acc(sess, accuracy_j, x_j, y_j, mnist.validation.images, mnist.validation.labels, N_valid, True)
                    # valid_acc_m = evaluate_acc(sess, accuracy_m, x_m, y_m, mnist.validation.images, mnist.validation.labels, N_valid, False)
                    # print("Valid accuracy on joint: {0}".format(valid_acc_j))
                    # print("Valid accuracy on marginals: {0}".format(valid_acc_m))

            # Evaluate mutual information on test set for each trial
            # 3 trials means 3 test_mi for each subset partition
            test_mi = evaluate_mi(sess, f_j_mean, x_j, y_j, X_test, y_test, N_test)
            # print("Test MI for {0} of the data: {1}".format(train_subset[k], test_mi))
            f_mi.write(str(k) + str("test mi:" + str(test_mi) + "\n"))

        f_mi.close()

