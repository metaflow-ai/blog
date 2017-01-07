import os
import tensorflow as tf

dir = os.path.dirname(os.path.realpath(__file__))

# First, you design your mathematical operations
# We are the default graph scope

# Let's design a variable
v1 = tf.Variable(1. , name="v1")
v2 = tf.Variable(2. , name="v2")
# Let's design an operation
a = tf.add(v1, v2)

# Let's create a Saver object
# By default, the Saver handles every Variables related to the default graph
all_saver = tf.train.Saver() 
# But you can precise which vars you want to save under which name
v2_saver = tf.train.Saver({"v2": v2}) 

# By default the Session handles the default graph and all its included variables
with tf.Session() as sess:
  # Init v and v2   
  sess.run(tf.global_variables_initializer())
  # Now v1 holds the value 1.0 and v2 holds the value 2.0
  # We can now save all those values
  all_saver.save(sess, dir + '/results/data.chkp')
  # or saves only v2
  v2_saver.save(sess, dir + '/results/data-v2.chkp')