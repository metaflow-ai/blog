import os, itertools
import tensorflow as tf
from tensorflow.models.rnn.ptb import reader

dir = os.path.dirname(os.path.realpath(__file__))

# Let's take a usual usecase you might have:
# You want to train a NN on a NLP task but first you want to train an embedding on your corpus to maximise your result (arguing about the fact that word embedding are better/worst than character solutions is beyond the scope of this article)
# This lead you to create

# here is our corpus
text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Pellentesque id venenatis lacus, in malesuada arcu. Donec nec tortor vitae ipsum convallis tincidunt. Integer molestie vestibulum sem, ut porta felis rutrum at. Cum sociis natoque penatibus et magnis dis parturient montes, nascetur ridiculus mus. Suspendisse sit amet sapien quam. Aliquam erat volutpat. Nunc sagittis arcu ac posuere eleifend. Cras ac orci dapibus, accumsan sapien vitae, dignissim ante. Nam eu ipsum nec eros sollicitudin lobortis in et enim. In placerat, ante vitae pellentesque bibendum, dolor quam fermentum justo, vitae pellentesque nisi ante sed arcu. Donec dictum hendrerit sodales."
corpus = text.split()
corpus_length = len(corpus)
# let's create an embedding
# first let's build our dictrionnariesa nd count how namy unique word (token) we have:
tokens = set(corpus)
word_to_id_dict = { word:id for id, word in enumerate(tokens) }
id_to_word_dict = { id:word for word,id in word_to_id_dict.items() }
id_corpus = [ word_to_id_dict[word] for word in corpus ]
nb_token = len(tokens)

# This will be our first dimension, then we will choose (this is an hyper parameter) in how many dimensions we want to mebed our words
# For the sake of this blog post, let's keep it very simple and chose 20
dim_embedding = 1000

# We will train this embedding with the most simple prior we can: a word is dependant on the previous one
# So the input of our model is a word and the output is the id of next word
with tf.variable_scope("placeholder"):
    x = tf.placeholder(tf.int32, shape=[None, None], name="x")
    y_true = tf.placeholder(tf.int32, shape=[None, None], name="y_true")

with tf.variable_scope("embedding"):
    # Let's build our embedding, we intialize it with s impel random normal centerd on 0 with a small variance
    embedding = tf.get_variable("embedding", shape=[nb_token, dim_embedding], initializer=tf.random_normal_initializer(0., 1e-3))

    # Then we retrieve the context word
    word_vectors = tf.nn.embedding_lookup(embedding, x, name="lookup") # Dim: bs x 1 x dim_embedding
    word_vectors = tf.squeeze(word_vectors) # Dim: bs x dim_embedding

    # Now we calculated the dot product of the current words with all other words
    z = tf.matmul(word_vectors, tf.transpose(embedding))

with tf.variable_scope("loss"):
    y_true_reshaped = tf.reshape(y_true, [-1])
    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(z, y_true_reshaped)
    loss_op = tf.reduce_mean(losses)

    # Let's track the loss
    tf.scalar_summary('loss', loss_op)

with tf.variable_scope("Accuracy"):
    a = tf.nn.softmax(z)
    predictions = tf.cast(tf.argmax(a, 1, name="predictions"), tf.int32)
    correct_predictions = tf.equal(predictions, y_true_reshaped)
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name="accuracy")

with tf.variable_scope('Optimizer'):
    adam = tf.train.AdamOptimizer(1e-3)
    global_step_tensor = tf.Variable(0, name="global_step", trainable=False)
    train_op = adam.minimize(loss_op, global_step=global_step_tensor, name="train_op")

# Since we did not defined a graph before defining our operations
# They all went in the default graph which TF create for us before hand
graph = tf.get_default_graph()

# We are going to use a helper function to create our inputs and labels
# The ptb_iterator is just a helper shifiting all the words by one and returning
# seq_length words for you to predict the next output
# very handy for RNN or our case
seq_length = 1 # We decided to predict the next word thanks only to its previous word
nb_epochs = 200
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())

    for i in range(nb_epochs):
        input_gen = reader.ptb_iterator(id_corpus, corpus_length // 6, seq_length)
        for x_batch, y_true_batch in input_gen:
            to_compute = [train_op, loss_op, global_step_tensor]
            feed_dict = {
                x: x_batch,
                y_true: y_true_batch
            }
            _, loss, global_step = sess.run(to_compute, feed_dict=feed_dict)

        if i % 10 == 0:
            print('Epoch %d/%d - loss:%f' % (i, nb_epochs, loss))

    # We compute the final accuracy
    x_batch = []
    y_batch = []
    for i, word in enumerate(id_corpus):
        if i == len(id_corpus) - 1:
            continue
        x_batch.append([id_corpus[i]])
        y_batch.append([id_corpus[i+1]])
    feed_dict = {
        x: x_batch,
        y_true: y_batch
    }
    acc = sess.run(accuracy, feed_dict=feed_dict)
    print('Final accuracy: %f' % acc)
# input_gen = reader.ptb_iterator(id_corpus, corpus_length // 2, seq_length)
# input_gen = itertools.cycle(input_gen)
# def feed_function():
#     x_batch, y_true_batch = next(input_gen)
#     return {
#         x: x_batch,
#         y_true: y_true_batch
#     }

# final_loss = tf.contrib.learn.train(
#     graph,
#     dir + "/results",
#     train_op,
#     loss_op,
#     global_step_tensor=global_step_tensor,
#     log_every_steps=10,
#     steps=nb_epochs,
#     supervisor_save_model_secs=1,
#     keep_checkpoint_max=5,
#     supervisor_save_summaries_steps=1,
#     feed_fn=feed_function
# )

