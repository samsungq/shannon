import tensorflow as tf


# Variable initializer                                                                                                    
initializer = tf.truncated_normal_initializer(stddev=0.02)

################################################################################################
# MNIST
################################################################################################

# MLP network (5 layers)
def mlp5(x, y):
    with tf.variable_scope("fc1"):
        h1 = tf.layers.dense(x, 200, activation=tf.nn.relu, kernel_initializer=initializer)

    with tf.variable_scope("fc2"):
        h2 = tf.layers.dense(h1, 100, activation=tf.nn.relu, kernel_initializer=initializer)

    with tf.variable_scope("fc3"):
        h3 = tf.layers.dense(h2, 10, activation=tf.nn.relu, kernel_initializer=initializer)

    # Concatenate inputs\
    xy = tf.concat([h3, y], axis=1)

    with tf.variable_scope("fc4"):
        h4 = tf.layers.dense(xy, 50, activation=tf.nn.relu, kernel_initializer=initializer)

    with tf.variable_scope("fc5"):
        h5 = tf.layers.dense(h4, 1, activation=None, kernel_initializer=initializer)

    return h5


# CNN network (6 layers)
def cnn6(x, y):
    x = tf.reshape(x, [-1, 28, 28, 1])

    with tf.variable_scope("conv1"):
        h1 = tf.layers.conv2d(x, 32, 5, padding="SAME", activation=tf.nn.relu, kernel_initializer=initializer)
        h1 = tf.layers.max_pooling2d(h1, 2, 2)

    with tf.variable_scope("conv2"):
        h2 = tf.layers.conv2d(h1, 64, 5, padding="SAME", activation=tf.nn.relu, kernel_initializer=initializer)
        h2 = tf.layers.max_pooling2d(h2, 2, 2)

    with tf.variable_scope("fc3"):
        h3 = tf.layers.flatten(h2)
        h3 = tf.layers.dense(h3, 256, activation=tf.nn.relu, kernel_initializer=initializer)

    with tf.variable_scope("fc4"):
        h4 = tf.layers.dense(h3, 10, activation=tf.nn.relu, kernel_initializer=initializer)    

    # Concatenate inputs
    xy = tf.concat([h4, y], axis=1)

    with tf.variable_scope("fc5"):
        h5 = tf.layers.dense(xy, 50, activation=tf.nn.relu, kernel_initializer=initializer)

    with tf.variable_scope("fc6"):
        h6 = tf.layers.dense(h5, 1, activation=None, kernel_initializer=initializer)

    return h6

