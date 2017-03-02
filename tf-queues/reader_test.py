import os, time
import tensorflow as tf

dir = os.path.dirname(os.path.realpath(__file__))

filenames = tf.train.match_filenames_once(dir + '/*.txt')
filename_queue = tf.train.string_input_producer(filenames)
reader = tf.WholeFileReader()
key, value = reader.read(filename_queue)

splitted_textfile = tf.string_split([value], " ")
value_size = tf.size(splitted_textfile.values)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    print(sess.run(value_size))

    coord.request_stop()
    coord.join(threads)