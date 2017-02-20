import tensorflow as tf
import numpy as np


with tf.variable_scope('Placeholder'):
    inputs_placeholder = tf.placeholder(tf.float32, name='inputs_placeholder', shape=[None, 10])
    labels_placeholder = tf.placeholder(tf.float32, name='labels_placeholder', shape=[None, 1])

with tf.variable_scope('NN'):
    W1 = tf.get_variable('W1', shape=[10, 1], initializer=tf.random_normal_initializer(stddev=1e-1))
    b1 = tf.get_variable('b1', shape=[1], initializer=tf.constant_initializer(0.1))
    W2 = tf.get_variable('W2', shape=[10, 1], initializer=tf.random_normal_initializer(stddev=1e-1))
    b2 = tf.get_variable('b2', shape=[1], initializer=tf.constant_initializer(0.1))

    a = tf.nn.relu(tf.matmul(inputs_placeholder, W1) + b1)
    a2 = tf.nn.relu(tf.matmul(inputs_placeholder, W2) + b2)

    y = tf.divide(tf.add(a, a2), 2)

with tf.variable_scope('Loss'):
    loss = tf.reduce_sum(tf.square(y - labels_placeholder) / 2)

with tf.variable_scope('Accuracy'):
    predictions = tf.greater(y, 0.5, name="predictions")
    correct_predictions = tf.equal(predictions, tf.cast(labels_placeholder, tf.bool), name="correct_predictions")
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))


adam = tf.train.AdamOptimizer(learning_rate=1e-3)
train_op = adam.minimize(loss)

# generate_data
inputs = np.random.choice(10, size=[10000, 10])
labels = (np.sum(inputs, axis=1) > 45).reshape(-1, 1).astype(np.float32)
print('inputs.shape:', inputs.shape)
print('labels.shape:', labels.shape)


test_inputs = np.random.choice(10, size=[100, 10])
test_labels = (np.sum(test_inputs, axis=1) > 45).reshape(-1, 1).astype(np.float32)
print('test_inputs.shape:', test_inputs.shape)
print('test_labels.shape:', test_labels.shape)

batch_size = 32
epochs = 10

batches = []
print("%d items in batch of %d gives us %d full batches and %d batches of %d items" % (
    len(inputs),
    batch_size,
    len(inputs) // batch_size,
    batch_size - len(inputs) // batch_size,
    len(inputs) - (len(inputs) // batch_size) * 32)
)
for i in range(len(inputs) // batch_size):
    batch = [ inputs[batch_size*i:batch_size*i+batch_size], labels[batch_size*i:batch_size*i+batch_size] ]
    batches.append(list(batch))
if (i + 1) * batch_size < len(inputs):
    batch = [ inputs[batch_size*(i + 1):],labels[batch_size*(i + 1):] ]
    batches.append(list(batch))
print("Number of batches: %d" % len(batches))
print("Size of full batch: %d" % len(batches[0]))
print("Size if final batch: %d" % len(batches[-1]))

global_count = 0
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(epochs):
        for batch in batches:
            # print(batch[0].shape, batch[1].shape)
            train_loss , _= sess.run([loss, train_op], feed_dict={
                inputs_placeholder: batch[0],
                labels_placeholder: batch[1]
            })
            # print('train_loss: %d' % train_loss)

            if global_count % 100 == 0:
                acc = sess.run(accuracy, feed_dict={
                    inputs_placeholder: test_inputs,
                    labels_placeholder: test_labels
                })
                print('accuracy: %f' % acc)
            global_count += 1

    acc = sess.run(accuracy, feed_dict={
        inputs_placeholder: test_inputs,
        labels_placeholder: test_labels
    })
    print("final accuracy: %f" % acc)

    saver = tf.train.Saver()
    last_chkp = saver.save(sess, 'results/graph.chkp')

for op in tf.get_default_graph().get_operations():
    print(op.name)
