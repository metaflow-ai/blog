# Tensor shapes in TensorFlow and dynamic batch size

Here is a simple HowTo to understand the concept of shapes in TensorFlow and hopefully avoid losing hours of debugging them.

### What is a tensor?
Very briefly:

A tensor is an array of n-dimension containing the same type of data (int32, bool, etc.)

A tensor can be described with what we call a shape: it is a list (or tuple) of numbers describing the size of each dimension of our tensor, for example:

A shape is a list (or tuple) of numbers describing the size of the array in each dimension, for exemple (D_*, W, H are integers):
- For a tensor of n dimensions: **(D_0, D_1, …, D_n-1)**
- For a tensor of size **W x H** (usually called a matrix): **(W, H)**
- For a tensor of size **W** (usually called a vector): **(W,)**
- For a simple scalar (those are equivalent): **() or (1,)**
*Note: (**D_***, **W** and **H** are integers)*

* Note on the vector (1-D tensor): it is impossible to determine if a vector is a row or column vector by looking at the vector shape in TensorFlow, and in fact it doesn’t matter. For more information please look at this stack overflow answer about NumPy notation ( which is roughly the same as TensorFlow notation): http://stackoverflow.com/questions/22053050/difference-between-numpy-array-shape-r-1-and-r *

A tensor looks like this in TensorFlow:
```python
my_tensor = tf.constant(0., shape=[6,3,7])
print(my_tensor) # -> Tensor("Const_1:0", shape=(6, 3, 7), dtype=float32)
```
We can see we have a Tensor object:
- It has a **name** used in a key-value store to retrieve it later: **Const:0**
- It has a **shape** describing the size of each dimension: **(6, 3, 7)**
- It has a **type**: **float32**

Now, here is the the most important piece of this article: **Tensor in TensorFlow has 2 shapes**

**The static shape AND the dynamic shape**

### The static shape
The static shape is the shape you provided when creating a tensor OR the shape inferred by TensorFlow when you define an operation resulting in a new tensor. It is a tuple or a list.

TensorFlow will do its best to guess the shape of your different tensors (between your different operations) but it won’t always be able to do it. Especially if you start to do operations with placeholder defined with unknown dimensions (like when you want to use a dynamic batch size).

To use the static shape (Accessing/changing) in your code, you will use the different functions which are **attached to the Tensor itself and have an underscore in their names**:
```python
# I create a first Tensor with a defined shape
my_tensor = tf.constant(0, shape=[6,2])
my_static_shape = my_tensor.get_shape() 
# -> TensorShape([Dimension(6), Dimension(2)])

print(my_static_shape) 
# -> (6, 2)
# You can get it as a list too
print(my_static_shape.as_list()) 
# -> [6, 2]

# I do an operation resulting in a new tensor
my_tensor_transposed = tf.transpose(my_tensor)
print(my_tensor_transposed.get_shape()) 
# -> TensorShape([Dimension(2), Dimension(6)])
# This shape has been inferred by TensorFlow based on the transpose operation 

# At any moment, you can also force a Tensor to have a precise shape

```

*Note: The static shape is very useful to debug your code with print so you can check your tensors have the right shapes.*

### The dynamic shape
The dynamic shape is the actual one used when you `run` your graph. It is itself a tensor describing the shape of the original tensor.

If you defined a placeholder with undefined dimensions (with the `None` type as a dimension), those `None` dimensions will only have a real value when you feed an input to your placeholder and so forth, any variable depending on this placeholder.

To use the dynamic shape(Accessing/changing) in your code, you will use the different functions which **are attached to the main scope and don’t have an underscore in their names**:
```python
my_tensor = tf.constant(0, shape=[6,2]) # <tf.Tensor 'Const_4:0' shape=(6, 2) dtype=int32>
my_dynamic_shape = tf.shape(my_tensor) # <tf.Tensor 'Shape:0' shape=(2,) dtype=int32>
# The shape is (2,) because my_tensor is a 2-D tensor, so the dynamic shape is a 1-D tensor containing size of my_tensor dimensions

my_reshaped_tensor = tf.reshape(my_tensor, [2, 3, 2]) # <tf.Tensor 'Reshape_2:0' shape=(2, 3, 2) dtype=int32>

# To access a dynamic shape value, you need to run your graph and feed any placeholder that your tensor my depended upon:
print(my_dynamic_shape.eval(session=tf.Session(), feed_dict={...}))
```

*The dynamic shape is very handy for dealing with dimensions that you want to keep dynamic.*

## The RNN use case
So here we are, we like dynamic inputs because we want to build a dynamic RNN which should be able to handle any different length of inputs.

In the training phase we will define a placeholder with a dynamic batch_size, and then we will use the TensorFlow API to create a LSTM. You will end up with something like this:
```python
my_placeholder = tf.placeholder(tf.float32, shape=[None, seq_max_length])

cell = tf.nn.rnn_cell.LSTMCell(num_units=50, state_is_tuple=True)

init_state = cell.zero_state(batch_size=???, tf.float32)
outputs, encoder_final_state = tf.nn.dynamic_rnn(cell, rnn_inputs, initial_state=init_state)
```
And now you need to initialize the init_state with `init_state = cell.zero_state(batch_size, tf.float32)` ...

But what the batch_size input should be equal to? Remember you want it to be dynamic so what are our options? TensorFlow allows different types here, if you read the source code you will find:
```python
Args:
      batch_size: int, float, or unit Tensor representing the batch size.
```
**int** and **float** can’t be used because when you define your graph, you actually don’t know what the batch_size will be (that’s the point).

The interesting piece is the last type: **"unit Tensor representing the batch size"**

The interesting piece is the last type: **“unit Tensor representing the batch size”**. If you dig the doc up from there, you will find that a unit Tensor is a “0-d Tensor” which is just a scalar. So how do you get that scalar-tensor anyway?

If you try with the static shape:
```python
batch_size = my_tensor.get_shape()[0] # Dimension(None)
print(batch_size)
# -> ? 
```
`batch_size` will be the `Dimension(None)` type (printed as ‘?’). This type can only be used as a dimension for placeholders.

What you actually want to do is to keep the dynamic `batch_size` “flow” though the graph, so you must use the dynamic shape:
```python
batch_size = tf.shape(my_tensor)[0] # <tf.Tensor 'strided_slice:0' shape=() dtype=int32>
print(batch_size)
# -> Tensor("strided_slice:0", shape=(), dtype=int32)
```
`batch_size` will be a TensorFlow `0-d Tensor (Scalar Tensor)` type describing the batch dimension, hooray!

## Conclusion
- Use the static shape for debugging
- Use the dynamic shape everywhere else especially when you have undefined dimensions

*Remark: In the RNN API, Tensorflow is taking care of the init_state and initiliaze it to the zero_state, why would i need to manually define it this way?*

You might want to control the initialization of the init_state when you run your graph. By having access to the variable init_state this way, we can do it because when you run a graph, **you can actually use the feed_dict to feed any variable at hand in your graph!**

```python
sess.run(encoder_final_state, feed_dict={
  init_state: previous_encoder_final_state
})
```
And now you can predict words Ad vitam æternam, **Cheers!** :beer:


## References
*Dive deep by reading the doc and the code!*

[https://www.tensorflow.org/versions/r0.11/resources/faq.html](TensorFlow faq)

[https://www.tensorflow.org/versions/r0.11/resources/dims_types.html](TensorFlow dimensions types documentation)
