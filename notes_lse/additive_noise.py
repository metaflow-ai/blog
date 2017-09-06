import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math

import tensorflow as tf

#############################################
# The global process is as follow:
# 1 - We fix an interval: [-2pi, 2pi]
# 2 - We fix a distribution for our data: uniform distribution
# 2 - We sample some input points from the distribution on the interval 
# 3 - We compute their corresponding sinus value
# 4 - We add additive noise to our perfect output values to create our noisy output values
# 5 - We learn from the dataset we juste created and recover the sinus function on the interval
# 6 - We show that if we sample some data outside of the interval, mothing works anymore 
#       (breaking the identically distributed assumption)
#############################################

#### PART 0: Gather a dataset
# We fix the standard deviation of our stochastic noise
# Remember that this value fix, on average, the lower bound for our loss
# In our case, the loss should vary around 0.5 * 1.^2 = 0.5 (Look at the loss)
noise_std_dev = 1. 

# We generate our training set: 100 000 points 
# The training data are I.I.D and following an additive noise pattern
nb_points = 100000
nb_val_points = 10000
batch_size = 512
epoch_size = (nb_points - nb_val_points) // batch_size
nb_epochs = 200
x = np.random.uniform(-2*math.pi, 2*math.pi, [nb_points, 1])
noisy_y = np.sin(x) + np.random.normal(scale=noise_std_dev, size=(nb_points, 1))
train_x = x[:nb_points - nb_val_points]
train_y = noisy_y[:nb_points - nb_val_points]
val_x = x[nb_points - nb_val_points:]
val_y = noisy_y[nb_points - nb_val_points:]
#### END PART 0

#### PART 1: Explore visually your dataset and gather insights
# Let's graph an example of our dataset
# We draw points uniformly from the interval
lin_nb_points = 100
lin_x = np.linspace(-2*math.pi, 2*math.pi, lin_nb_points).reshape(-1, 1) # 10x1

# We retrieve the sinus values
lin_y = np.sin(lin_x)

lin_noises = np.random.normal(scale=noise_std_dev, size=(lin_nb_points, 1))
# ... and add it to the perfect sinus values
lin_noisy_y = lin_y + lin_noises

f, axarr = plt.subplots(4, sharex=True)
f.subplots_adjust(hspace=1.)
axarr[0].set_title('Sinus function: what we want to approximate')
axarr[0].plot(lin_x, lin_y)
axarr[1].set_title('Noisy Sinus data: what we actually observe')
axarr[1].scatter(val_x, val_y, s=.1)
#### END PART 1

#### PART 2: Build a model based on the insights we gathered
# We will build a basic 2-layer FC NN (It is an universal approximator after all)
nb_hd_units = 50
with tf.variable_scope('model'):
    x_plh = tf.placeholder(tf.float32, name='x_plh', shape=[None, 1])
    y_plh = tf.placeholder(tf.float32, name='y_plh', shape=[None, 1])

    # Notice that I use the uniform prior to initialize my parameters
    # In this case: Maximum a posteriori (MAP) = Maximum Likelihood Estimate (MLE)
    w1 = tf.get_variable('w1', shape=[1, nb_hd_units], 
        initializer=tf.random_uniform_initializer(-2e-1, 2e-1))
    b1 = tf.get_variable('b1', shape=[nb_hd_units], 
        initializer=tf.random_uniform_initializer(0, 1e-2))
    a = tf.nn.relu(tf.matmul(x_plh, w1) + b1)

    w2 = tf.get_variable('w2', shape=[nb_hd_units, nb_hd_units], 
        initializer=tf.random_uniform_initializer(-2e-1, 2e-1))
    b2 = tf.get_variable('b2', shape=[nb_hd_units], 
        initializer=tf.random_uniform_initializer(0, 1e-2))
    a = tf.nn.relu(tf.matmul(a, w2) + b2)

    w3 = tf.get_variable('w3', shape=[nb_hd_units, 1], 
        initializer=tf.random_uniform_initializer(-2e-1, 2e-1))
    out = tf.matmul(a, w3)

with tf.variable_scope('loss'):
    # From the optimization point of view, I need a decreasing learning rate to converge
    global_step = tf.Variable(0, trainable=False)
    starter_lr = 1e-3
    # See this as: the first half of training is to fit globally the data, 
    # the last half of the training is to make more fine grained tuning
    lr = tf.train.exponential_decay(starter_lr, global_step, nb_epochs // 2 * epoch_size, 0.3, staircase=True)

    # I'm keeping this simple:
    # - LSE
    # - No regularization terms are added
    # - The basic gradient descent, nothing fancy like Adam
    loss = 1/2 * tf.reduce_sum(tf.square(out - y_plh))
    sgd = tf.train.GradientDescentOptimizer(lr)
    train_op = sgd.minimize(loss, global_step=global_step)
#### END PART 2

#### PART 3: Training and graphing accuracy
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # We train our model
    for i in range(nb_epochs): # 10 epochs
        # To use stochastic approximation, we shuffle our training set for each epoch
        perm = np.random.permutation(len(train_x))
        perm_x = train_x[perm]
        perm_noisy_y = train_y[perm]

        for j in range(epoch_size):
            _, l, lr_out = sess.run([train_op, loss, lr], feed_dict={
                x_plh: perm_x[j*batch_size:(j+1)*batch_size, :],
                y_plh: perm_noisy_y[j*batch_size:(j+1)*batch_size, :]
            })

        if i % 10 == 0:
            val_loss = sess.run(loss, feed_dict={
                x_plh: val_x,
                y_plh: val_y
            })
            print("Epoch: %d - %d%%, lr: %f, mean_val_loss: %f" % (i, int((j + 1) / epoch_size * 100), lr_out, val_loss / nb_val_points))

    # Let's graph the output of our neural network on the same distribu
    lin_neural_y = sess.run(out, feed_dict={
        x_plh: lin_x
    })
    errors = np.abs(lin_neural_y - lin_y)

    # One last thing:
    # If I Sample some data points outside of the training interval, 
    # meaning I'm sampling data without an identical distribution
    # Clearly, our model is spitting out ridiculous values.
    lin_x_not_iid = np.linspace(0, 4*math.pi, lin_nb_points).reshape(-1, 1)
    lin_neural_y_not_iid = sess.run(out, feed_dict={
        x_plh: lin_x_not_iid
    })

axarr[2].set_title('Neural reconstruction, MSE: ' + str(int(np.mean(np.square(errors)) * 1000)/1000))
axarr[2].plot( lin_x, lin_neural_y )
axarr[3].set_title('Test set translated by 2 pi (not IID)')
axarr[3].plot( lin_x_not_iid, lin_neural_y_not_iid )
plt.savefig('sinus.png')