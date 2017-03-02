import os
import numpy as np
import tensorflow as tf

dir = os.path.dirname(os.path.realpath(__file__))

# We simulates some raw inputs data
# let's say we receive 100 batches, each containing 50 elements
x_inputs_data = tf.random_normal([2], mean=0, stddev=1)
# q = tf.FIFOQueue(capacity=10, dtypes=tf.float32)
# enqueue_op = q.enqueue_many(x_inputs_data)

# input = q.dequeue()

# numberOfThreads = 1
# qr = tf.train.QueueRunner(q, [enqueue_op] * numberOfThreads)

batch_input = tf.train.batch(
    [x_inputs_data], 
    batch_size=3, 
    num_threads=1, 
    capacity=32, 
    enqueue_many=False, 
    shapes=None, 
    dynamic_pad=False, 
    allow_smaller_final_batch=True
)
with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    # threads = qr.create_threads(sess, coord=coord, start=True)

    print(sess.run([x_inputs_data, batch_input]))
    coord.request_stop()
    coord.join(threads)