import tensorflow as tf

q = tf.FIFOQueue(capacity=3, dtypes=tf.float32)

x_input_data = tf.random_normal([3], mean=-1, stddev=4)
enqueue_many_op = q.enqueue_many(x_input_data) # <- x1 - x2 -x3 |
enqueue_op = q.enqueue(x_input_data) # <- [x1, x2, x3] - void - void |

x_input_data2 = tf.random_normal([3, 1], mean=-1, stddev=4)
enqueue_many_op2 = q.enqueue_many(x_input_data2) # <- [x1] - [x2] - [x3] |
enqueue_op2 = q.enqueue(x_input_data2) # <- [ [x1], [x2], [x3] ] - void - void |

dequeue_op = q.dequeue() 

with tf.Session() as sess:
    sess.run(enqueue_many_op)
    print(sess.run(dequeue_op), sess.run(dequeue_op), sess.run(dequeue_op))
    
    sess.run(enqueue_op)
    print(sess.run(dequeue_op))

    sess.run(enqueue_many_op2)
    print(sess.run(dequeue_op), sess.run(dequeue_op), sess.run(dequeue_op))

    sess.run(enqueue_op2)
    print(sess.run(dequeue_op))