import time, os, argparse, io

import tensorflow as tf
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
  
dir = os.path.dirname(os.path.realpath(__file__))

# Note: elu is not bounded, yet it works
def univAprox(x, N=50, phi=tf.nn.elu, reuse=False): # First trick: the reuse capacity
    with tf.variable_scope('UniversalApproximator', reuse=reuse):
        x = tf.expand_dims(x, -1)

        # Second trick: using convolutions!
        aW_1 = tf.get_variable('aW_1', shape=[1, 1, N], initializer=tf.random_normal_initializer(stddev=.1))
        ab_1 = tf.get_variable('ab_1', shape=[N], initializer=tf.constant_initializer(0.))
        z = tf.nn.conv1d(x, aW_1, stride=1, padding='SAME') + ab_1
        a = phi(z)

        aW_2 = tf.get_variable('aW_2', shape=[1, N, 1], initializer=tf.random_normal_initializer(stddev=.1))
        z = tf.nn.conv1d(a, aW_2, stride=1, padding='SAME')

        out = tf.squeeze(z, [-1])
    return out

def func_to_approx(x):
    return tf.sin(x)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--nb_neurons", default=50, type=int, help="Number of neurons")
    args = parser.parse_args()

    with tf.variable_scope('Graph') as scope:
        # Graph
        x = tf.placeholder(tf.float32, shape=[None, 1], name="x")
        y_true = func_to_approx(x)
        y = univAprox(x, args.nb_neurons)
        loss = tf.reduce_mean(tf.square(y - y_true))
        loss_summary_t = tf.summary.scalar('loss', loss)
        adam = tf.train.AdamOptimizer(learning_rate=1e-2)
        train_op = adam.minimize(loss)

    # Plot graph
    img_strbuf_plh = tf.placeholder(tf.string, shape=[])
    my_img = tf.image.decode_png(img_strbuf_plh, 4)
    img_summary_t = tf.summary.image('img', tf.expand_dims(my_img, 0))

    saver = tf.train.Saver()
    with tf.Session() as sess:
        result_folder = dir + '/results/' + str(int(time.time()))
        sw = tf.summary.FileWriter(result_folder, sess.graph)

        print('Training our universal approximator')
        sess.run(tf.global_variables_initializer())
        for i in range(2000):
            x_in = np.random.uniform(-10, 10, [100000, 1])

            current_loss, loss_summary, _ = sess.run([loss, loss_summary_t, train_op], feed_dict={
                x: x_in
            })
            sw.add_summary(loss_summary, i + 1)

            if (i + 1) % 100 == 0:
                print('batch: %d, loss: %f' % (i + 1, current_loss))

        print('Plotting graphs')
        inputs = np.array([ [(i - 1000) / 100] for i in range(2000) ])
        y_true_res, y_res = sess.run([y_true, y], feed_dict={
            x: inputs
        })
        plt.figure(1)
        plt.subplot(211)
        plt.plot(inputs, y_true_res.flatten())
        plt.subplot(212)
        plt.plot(inputs, y_res)
        imgdata = io.BytesIO()
        plt.savefig(imgdata, format='png')
        imgdata.seek(0)
        img_summary = sess.run(img_summary_t, feed_dict={
            img_strbuf_plh: imgdata.getvalue()
        })
        sw.add_summary(img_summary, i + 1)
        plt.clf()

        # Saving the graph
        saver.save(sess, result_folder + '/data.chkp')
        