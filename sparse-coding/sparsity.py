import time, os

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

dir = os.path.dirname(os.path.realpath(__file__))
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# Fully connected model
# Number of parameters: (784 * 784 + 784) + (784 * 10 + 10) = 615440 + 7850 = 623290
# Dimensionality: R^784 -> R^784 -> R^10

# Placeholder
x = tf.placeholder(tf.float32, shape=[None, 784])
y_true = tf.placeholder(tf.float32, shape=[None, 10])

sparsity_constraint = tf.placeholder(tf.float32)

# Variables
with tf.variable_scope('NeuralLayer'):
    W = tf.get_variable('W', shape=[784, 784], initializer=tf.random_normal_initializer(stddev=1e-1))
    b = tf.get_variable('b', shape=[784], initializer=tf.constant_initializer(0.1))

    z = tf.matmul(x, W) + b
    a = tf.nn.relu(z)

    # We graph the average density of neurons activation
    average_density = tf.reduce_mean(tf.reduce_sum(tf.cast((a > 0), tf.float32), reduction_indices=[1]))
    tf.scalar_summary('AverageDensity', average_density)

with tf.variable_scope('SoftmaxLayer'):
    W_s = tf.get_variable('W_s', shape=[784, 10], initializer=tf.random_normal_initializer(stddev=1e-1))
    b_s = tf.get_variable('b_s', shape=[10], initializer=tf.constant_initializer(0.1))

    out = tf.matmul(a, W_s) + b_s
    y = tf.nn.softmax(out)

with tf.variable_scope('Loss'):
    epsilon = 1e-7 # After some training, y can be 0 on some classes which lead to NaN 
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_true * tf.log(y + epsilon), reduction_indices=[1]))
    # We add our sparsity constraint on the activations
    loss = cross_entropy + sparsity_constraint * tf.reduce_sum(a)

    tf.scalar_summary('loss', loss) # Graph the loss

# We merge summaries before the accuracy summary to avoid 
# graphing the accuracy with training data
summaries = tf.merge_all_summaries()

with tf.variable_scope('Accuracy'):
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_true, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    acc_summary = tf.scalar_summary('accuracy', accuracy) 

# Training
adam = tf.train.AdamOptimizer(learning_rate=1e-3)
train_op = adam.minimize(loss)
sess = None
# We iterate over different sparsity constraint
for sc in [0, 1e-4, 5e-4, 1e-3, 2.7e-3]:
    result_folder = dir + '/results/' + str(int(time.time())) + '-fc-sc' + str(sc)
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        sw = tf.train.SummaryWriter(result_folder, sess.graph)
        
        for i in range(20000):
            batch = mnist.train.next_batch(100)
            current_loss, summary, _ = sess.run([loss, summaries, train_op], feed_dict={
                x: batch[0],
                y_true: batch[1],
                sparsity_constraint: sc
            })
            sw.add_summary(summary, i + 1)

            if (i + 1) % 100 == 0:
                acc, acc_sum = sess.run([accuracy, acc_summary], feed_dict={
                    x: mnist.test.images, 
                    y_true: mnist.test.labels
                })
                sw.add_summary(acc_sum, i + 1)
                print('batch: %d, loss: %f, accuracy: %f' % (i + 1, current_loss, acc))