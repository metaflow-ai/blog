import time, os, argparse, io

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from universal import univAprox

dir = os.path.dirname(os.path.realpath(__file__))
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# HyperParam
parser = argparse.ArgumentParser()
parser.add_argument("--nb_neurons", default=50, type=int, help="Number of neurons")
args = parser.parse_args()

with tf.variable_scope('Graph') as scope:
    # Graph
    x = tf.placeholder(tf.float32, shape=[None, 784], name="x")
    y_true = tf.placeholder(tf.float32, shape=[None, 10], name="y_true")

    W1 = tf.get_variable('W1', shape=[784, 200], initializer=tf.random_normal_initializer(stddev=1e-1))
    b1 = tf.get_variable('b1', shape=[200], initializer=tf.constant_initializer(0.1))
    z = tf.matmul(x, W1) + b1
    a = univAprox(z, args.nb_neurons)

    W2 = tf.get_variable('W2', shape=[200, 50], initializer=tf.random_normal_initializer(stddev=1e-1))
    b2 = tf.get_variable('b2', shape=[50], initializer=tf.constant_initializer(0.1))
    z = tf.matmul(a, W2) + b2
    a = univAprox(z, args.nb_neurons, reuse=True)

    W_s = tf.get_variable('W_s', shape=[50, 10], initializer=tf.random_normal_initializer(stddev=1e-1))
    b_s = tf.get_variable('b_s', shape=[10], initializer=tf.constant_initializer(0.1))
    logits = tf.matmul(a, W_s) + b_s
    
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(None, y_true, logits))
    tf.summary.scalar('loss', loss) # Graph the loss
    adam = tf.train.AdamOptimizer(learning_rate=1e-3)
    train_op = adam.minimize(loss)    

    # We merge summaries before the accuracy summary to avoid 
    # graphing the accuracy with training data
    summaries = tf.summary.merge_all()

    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y_true, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    acc_summary = tf.summary.scalar('accuracy', accuracy) 

    # Plot graph
    plot_x = tf.placeholder(tf.float32, shape=[None, 1], name="plot_x")
    plot_y = univAprox(plot_x, args.nb_neurons, reuse=True)
    img_strbuf_plh = tf.placeholder(tf.string, shape=[])
    my_img = tf.image.decode_png(img_strbuf_plh, 4)
    img_summary = tf.summary.image(
        'matplotlib_graph'
        , tf.expand_dims(my_img, 0)
    )

saver = tf.train.Saver()
with tf.Session() as sess:
    result_folder = dir + '/results/' + str(int(time.time()))
    sess.run(tf.global_variables_initializer())
    sw = tf.summary.FileWriter(result_folder, sess.graph)
    
    print('Training')
    for i in range(20000):
        batch = mnist.train.next_batch(200)
        current_loss, summary, _ = sess.run([loss, summaries, train_op], feed_dict={
            x: batch[0],
            y_true: batch[1]
        })
        sw.add_summary(summary, i + 1)

        if (i + 1) % 1000 == 0:
            acc, acc_sum = sess.run([accuracy, acc_summary], feed_dict={
                x: mnist.test.images, 
                y_true: mnist.test.labels
            })
            sw.add_summary(acc_sum, i + 1)
            print('batch: %d, loss: %f, accuracy: %f' % (i + 1, current_loss, acc))

    print('Plotting approximated function graph')
    inputs = np.array([ [(i - 500) / 100] for i in range(1000) ])
    plot_y_res = sess.run(plot_y, feed_dict={
        plot_x: inputs
    })
    plt.figure(1)
    plt.plot(inputs, plot_y_res)
    imgdata = io.BytesIO()
    plt.savefig(imgdata, format='png')
    imgdata.seek(0)
    plot_img_summary = sess.run(img_summary, feed_dict={
        img_strbuf_plh: imgdata.getvalue()
    })
    sw.add_summary(plot_img_summary, i + 1)
    plt.clf()

    # Saving the graph
    saver.save(sess, result_folder + '/data.chkp')