import tensorflow as tf

p = tf.placeholder(tf.float32, shape=[], name="p")
v2 = tf.Variable(2. , name="v2")
a = tf.add(p, v2)

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  # From the moment we initiliaze variables, until the end of the Session
  # We can access them
  print(sess.run(v2)) # -> 2.

  # On the other hand, intermediate variables has to be recalculated 
  # each time you want to access its value
  print(sess.run(a, feed_dict={p: 3})) # -> 5.

  # Even if calculated the value of a, it's no more accessible
  # the value of a has been freed off the memory
  print(sess.run(a)) # Error ...
