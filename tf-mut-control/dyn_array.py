import os
import tensorflow as tf

dir = os.path.dirname(os.path.realpath(__file__))

embed = tf.get_variable(
    "embed",
    shape=[0, 1],
    dtype=tf.float32,
    validate_shape=False # This shape will evolve, so we need to remove any TensorFlow optim here
)
word_dict = tf.Variable(
    initial_value=[], 
    name='word_dict', 
    dtype=tf.string,
    validate_shape=False,
    trainable=False
)
textfile = tf.placeholder(tf.string)

# Update word dict
splitted_textfile = tf.string_split(textfile, " ")
tmp_word_dict = tf.concat([word_dict, splitted_textfile.values], 0)
tmp_word_dict, word_idx, word_count = tf.unique_with_counts(tmp_word_dict)
assign_word_dict = tf.assign(word_dict, tmp_word_dict, validate_shape=False)
with tf.control_dependencies([assign_word_dict]):
    word_dict_value = word_dict.read_value()
    missing_nb_dim = tf.shape(word_dict_value)[0] - tf.shape(embed)[0]
    missing_nb_dim = tf.Print(missing_nb_dim, data=[missing_nb_dim, word_dict_value], message="missing_nb_dim, word_dict:", summarize=10)

# Update embed
def update_embed_func():
    new_columns = tf.random_normal([missing_nb_dim, 1], mean=-1, stddev=4)
    new_embed = tf.concat([embed, new_columns], 0)
    assign_op = tf.assign(embed, new_embed, validate_shape=False)
    return assign_op

should_update_embed = tf.less(0, missing_nb_dim)
assign_to_embed = tf.cond(should_update_embed, update_embed_func, lambda: embed)
with tf.control_dependencies([assign_to_embed]):
    # outputs = tf.identity(outputs)
    embed_value = embed.read_value()
    word_embed = tf.embedding_lookup_sparse(embed_value, word_idx)

persistent_sess = tf.Session()
persistent_sess.run(tf.global_variables_initializer())

persistent_sess.run(assign_to_embed, feed_dict={
    textfile: ["that is cool! that is awesome!"]
})
print(persistent_sess.run(tf.trainable_variables()[0]))
persistent_sess.run(assign_to_embed, feed_dict={
    textfile: ["this is cool! that was crazy"]
})
print(persistent_sess.run(tf.trainable_variables()[0]))



tf.summary.FileWriter(dir, persistent_sess.graph).flush()

# More discussion on https://github.com/tensorflow/tensorflow/issues/7782