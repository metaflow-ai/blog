import os, time
import tensorflow as tf

# We are fucked, the models folder has been kicked out of tensorflow 1.0 deps
# I can't import this handy reader anymore
# (For anyone interested in the source code you can find it here: https://github.com/tensorflow/models/blob/master/tutorials/rnn/ptb/reader.py)
# from tensorflow.models.rnn.ptb import reader

dir = os.path.dirname(os.path.realpath(__file__))

# ain't cool, but hey! Let's rebuild one and even better
# Let's make it to create an online RNN which adapt automatically to unseen words
# crazy shit you say ? Not so much my friend

# Let's say some backend is pulling some text files fom somewhere and dumpt them in the current folder
# We will use a regexp to scan them
filenames = tf.train.match_filenames_once(dir + '/*.txt') # One minor thing, this op needs to be initialized
# And then we will build a queue to load them once at a time
# Again: multiple threads, no waiting
filenames = tf.Print(filenames, [filenames], message="filename_queue: ")
filename_queue = tf.train.string_input_producer(filenames) # Here we build a queue
reader = tf.WholeFileReader()
key, value = reader.read(filename_queue) # Key holds the filename and value the content

# So now, we have an async queue to load and read our text files from the filesystem

# What we want now, is an other async job to prepare all the text files content
# in a good batched format for our GPU
# Remember the only goal of queues is to avoid starving the GPU
# The bottleneck could very well be the environment around the GPU and the GPU itself

# We will train with each batch containing ...
batch_size = 5 # ... 5 sequence of ...
seq_length = 5 # ... 5 words
# This leads us to find the epoch_size
splitted_textfile = tf.string_split([value], " ")
text_length = tf.size(splitted_textfile.values) 
batch_len = text_length // batch_size
text = tf.reshape(splitted_textfile.values[:batch_size * batch_len], [batch_size, batch_len])
epoch_size = (batch_len - 1) // seq_length
range_q = tf.train.range_input_producer(epoch_size, shuffle=False)
index = range_q.dequeue()
x = text[:, index * seq_length:(index + 1) * seq_length]
y = text[:, index * seq_length + 1:(index + 1) * seq_length + 1]
with tf.Session() as sess:
    # We initialize Variables
    # This is when "match_filenames_once" run the regexp
    sess.run(tf.global_variables_initializer())

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    for i in range(1):
        print(sess.run([x, y]))
    # print("File %s beginning with %s" % (sess.run(key), str(sess.run(value)[:30]) + "...") )
    # print("File %s beginning with %s" % (sess.run(key), str(sess.run(value)[:30]) + "...") )
    # print("File %s beginning with %s" % (sess.run(key), str(sess.run(value)[:30]) + "...") )
    # print("File %s beginning with %s" % (sess.run(key), str(sess.run(value)[:30]) + "...") )
    
    coord.request_stop()
    coord.join(threads)
