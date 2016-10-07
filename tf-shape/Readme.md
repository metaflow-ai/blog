# Tensor shapes in TensorFlow and dynamic batch size

If you don't want to struggle hours debugging your shapes in TensorFlow, here is a simple HowTo to understand the concept

Very briefly, a Tensor is an array of n-dimension containing the same type (int32, bool, etc.)
Any tensor in general can be describe with what we call a shape:

It is a list (or tuple) of numbers describing the size of the array in each dimension, for exemple:
- For a tensor of n dimensions: [D_0, D_1, ..., D_n-1] 
- For a tensor of size W x H (usually called a matrix): [W, H] 
- For a tensor of size W (usually called a vector): [W,]
- For a simple scalar: [] or [1,] (those are equivalent)

*Note on the vector: in tensorflow it is impossible to determine if a vector is a row or column vector by looking at the shape, and in fact it doesn't matter*

Now, here is the the most important piece of this article: Tensor in TensorFlow has **2** shapes:
- The static shape
- The dynamic shape

### The static shape
The static shape is the shape inferred by Tensorflow when you define your computational graph

Tensorflow will do its best to guess the shape of your different but it won't always be able to do it.
Especially if you have a placeholder with a non-defined dimension size. Like when you want to use a dynamic batch size

To use this shape (Accessing or changing), you will use the different functions which are attached to the tensor:
```python
my_tensor = tf.constant(0, shape=[6,2])
my_static_shape = my_tensor.get_shape() # TensorShape([Dimension(6), Dimension(2)])
print(my_static_shape) # -> (6, 2)
 # you can get it as a list too
print(my_static_shape.as_list()) # -> [6, 2]

my_tensor.set_shape([d_0, ...]) # It is used mostly to ensure you have a precise shape
```

*The static shape is very usefull to debug your code with print to check what your are doing while you define your graph.*

### The dynamic shape
The dynamic shape is the actual one used when you compute things.

For example, if you defined a placeholder with undefined dimension (with `None`), this `None` dimension will only have a value when you provide a value to your placeholder and for any variable depending on this placeholder

To use this shape(Accessing or changing), you will use the different functions which are attached to the main scope:
```python
my_tensor = tf.constant(0, shape=[6,2]) # <tf.Tensor 'Const_4:0' shape=(5, 2) dtype=int32>
tf.shape(my_tensor) # <tf.Tensor 'Shape:0' shape=(2,) dtype=int32>

my_reshaped_tensor = tf.reshape(my_tensor, [2, 3, 2]) # <tf.Tensor 'Reshape_2:0' shape=(2, 3, 2) dtype=int32>
```

*The dynamic shape is not very handy for debugging, but it is very handi for dealing with undefined dimension*

## The RNN use case
So here we are, interested in dynamic inputs we want to build a RNN which should be able to handle any different length of lists

In the training phase will we defined a placeholder with a dynamic batch_size:
```python
my_tensor = tf.get_variable(shape=[None, seq_max_length])
```

Later on We want to use the TensorFlow API for RNN:
```python
state_is_tuple = True
cell = tf.nn.rnn_cell.LSTMCell(self.state_size, use_peepholes=True, state_is_tuple=state_is_tuple)

init_state = cell.zero_state(batch_size tf.float32)
outputs, encoder_final_state = tf.nn.dynamic_rnn(cell, rnn_inputs, initial_state=init_state)
```
And now, you need to initialize the init_state cell `cell.zero_state(batch_size, tf.float32)` ...

But what batch_size should be equal to, when you want it to be dynamic ?
Tensorflow allows different type here, if you open the inner code you will find:
```python
Args:
      batch_size: int, float, or unit Tensor representing the batch size.
```

The interesting piece here is: "unit Tensor representing the batch size"
If you dig the doc up from there, you will find that a unit Tensor is a 0-d Tensor which is a Scalar
Usually the shape will be `shape=()`
https://www.tensorflow.org/versions/r0.11/resources/dims_types.html

So how do you get that anyway? Coming back to our problems,
If you try:
```python
batch_size = my_tensor.get_shape()[0]
```
batch_size will be the None type, The None type can only be used as a dimension for placeholders
tensorflow like to know dimensions to check before hand if things will go well

If you try:
```python
batch_size = my_tensor.get_shape().as_list()[0]
```
batch_size will be the TensorFlow Dimension type printed as '?'
Surprinsingly, you still won't always be able to use that either

What you want to do is actually to keep the dynamic batch_size flow trhough the graph
And the only reliable way i found to do that is this way:
```python
batch_size = tf.shape(my_tensor)[0]
```
batch_size will be a TensorFlow Tensor type


## References
*Dive deep by reading the doc and the code!*
[https://www.tensorflow.org/versions/r0.11/resources/faq.html](Tensorflow faq)

