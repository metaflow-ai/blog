# Sparse coding: what is it? Why use it? How to use it?

*Disclaimer:* I don't own a Ph.D in machine learning, yet I'm deeply passionate about the field and try to learn as much as I can about it.

While I'm learning a new subject, I found out that it was a very valuable exercise to write a comprehensive guide about my study. So I'm now sharing my progress looking for useful feedback while helping others reach understanding faster. 

I've been spending the last few days to understand what is sparse coding, Here are my results!

We won't dive hard in any theoretical math, this article is trying to explore the sparse coding notion and see their impact on neural networks: I prefer to experiment with a notion and implement some concrete hypothesis than just playin with the math, i believe that in ML, concrete experience seems more relevant than pure theory.

**Please, if you find any errors, contact me: I'll be glad to hear from you :)**

**The reader should already have some understanding of what is a neural networks before reading this**

## What is Sparse coding?

### The mathematical standpoint
Sparse coding is the study of algorithms which aim to learn a useful **sparse representation** of any given data. Each datum will then be encoded as a *sparse code*:
- The algorithm only needs input data to learn the sparse representation. This is very useful since you can apply it directly on any kind of data, it is called unsupervised learning
- It will automatically find the representation without loosing any information (As if one could automatically reveals the intrinsic atoms of one's data).

To do so, sparse coding algorithms try to satisfy two constraints at the same time:
- For a given datum as a vector **x**, it will try to learn a "useful" sparse representation as a vector **h** 
- For each representation as a vector **h**, it will try to learn a basis **D** to reconstruct the original datum as a vector **x**.

The mathematical representation of the general objective function for this problem speaks for itself:
![General equation of the sparse coding algorithm](https://gist.githubusercontent.com/morgangiraud/9268b95bc1debd2feac37414c035ba03/raw/sparse-coding-equation.jpg "Sparse coding equation")
where:
- **N** is the number of datum in the data
- **x_k** is the **k** given vector of the data
- **h_k** is the sparse representation of **x_k**
- **D** D is a the decoder matrix (more on that later)
- **lambda** is the coefficient of sparsity
- **C** is a given constant

The general form looks like we are trying to nest two optimization problem (The double "min" expression). I prefer to see this as two different optimization problem competing against each other to find the best middle ground possible:
- The first "min" is acting only on the left side of the sum trying to minimize the **reconstruction loss** of the input by tweaking **D**.
- The second "min" tries to promote sparsity by minimizing the L1-norm of the sparse representation **h**
Put simply, we are just trying to resolve a problem (in this case, reconstruction) while using the least possible amount of resource we can to store our data.

**Note on the constraint on D rows:**

If you don't apply any constraint on D rows you can actually minimize **h** as much as you want while making **D** big enough to compensate. We want to avoid this behavior as we are looking for vector containing actual zero value with only as few as possible non-zeros and big values.

The value of C doesn't really matter, in our case, we will choose **C = 1**


#### The vector space interpretation:
If you try to find a vector space to plot a representation of your data, you need to find a basis of vectors.

Given a number of dimensions, **sparse coding** tries to learn an **over-complete basis** to represent data efficiently. To do so, you must have provided at first enough dimensions to learn this **over-complete basis**. 

In real life, you just give more (or same amount) dimensions than the number of dimensions the original data are encoded in. For example, for images of size 28x28 with only 1 channel (gray scale), our space containing our input data has 28x28x1=784 dimension

**Why would you need it to be over-complete?**

An over-complete basis means redundancy in your basis, and vectors (while training) will be able to "compete" to represent data more efficiently. Also you are assured, you won't need all dimensions to represent any data point: In an over-complete basis, you always can set some dimensions to 0.  

Features will have multiple representation in the obtained basis, but by adding the sparsity constraint you can enforce a unique representation as each feature will get encoded with the most sparse representation.

### The biological standpoint
From the biological standpoint the notion of sparse code (and sparse coding) comes after the more general notion of neural code (and neural coding).

*Wikipedia says:*
```
Neural coding is concerned with characterizing the relationship between the stimulus and the individual or ensemble neuronal responses and the relationship among the electrical activity of the neurons in the ensemble.
```
Let's say you have N binary (for simplicity) neurons (which can be biological or virtual). Basically:
- You will feed some inputs to your neural networks which will give you an output in returns. 
- To compute this output, the neural network will strongly activate some neurons while some others won't activate at all. 
- The observation of those activations for the a given input is what we call the neural code and the quantity of activations on some data is called the **average activity ratio**
- Training your neurons to reach an acceptable neural code is what we call neural coding.

Now that we know what a neural code is we can start to guess what neural code could looks like. You could have only 4 big cases:
- No neuron activate at all 
- Only one neuron gets activated
- Less than half of the neurons get activated
- Half of the neurons get activated

Let's put aside the case where no neurons activate at all as this case is pretty dull.

### Local code
At one extreme of low average activity ratio are local codes: Only one neurons gets activated per feature of the stimulus. No overlap between neurons is possible: one neuron can't react to different features. 

Note that a complex stimulus might contain multiple feature (multiple neurons can fire at the same time for a complex stimulus)

### Dense code
The other extreme corresponds to dense code: half of the neurons get activated per feature.

Why half (average activity ratio of 0.5)?

Remember we are looking at neurons in the ensemble, for any given distribution of activated neurons with an activity ratio > 0.5, you could swap activated ones with the inactivated ones without losing information in the ensemble. Remember that all neurons are used to encode information, to the opposite of sparse code, one neurons doen't hold any specific information in itself. 

### Sparse code
Everything in between those two extremes is what we call sparse code. We take into account only a subset of neurons to encode information.

Sparce code can be seen as a good compromise between balancing all the different characteristics of neural codes: computation power, fault tolerance, interference, complexity ... . This is why in an evolutionary perspective, it make sens that the brain works only with sparse code.

*More on that in the why section.*

### Linking the two fields
**How does neural code relate to the mathematical definition of sparse coding?**

This two fields can be linked thanks to neural networks: You can interpret a neural network in terms of projections of information into vector spaces: Each layer of neurons will receive a representation of the data (in a vector space) from the previous layer and project it to another vector space.

**If you interpret a layer as a data projection, what can be interpreted as the basis of the vector space?**

It's the neurons per layer: each neurons would represent a vector. if a neuron is activated in a layer: it means that you need this dimension to encode your data. The less neurons is activated per projection, the closer you are from a sparse representation of your data and a sparse code in the biological standpoint.

Choosing the number of neurons per layer is then equivalent as choosing the number of dimension of your over-complete basis which will represent your data. It is the same idea as word embedding when you choose the number of dimensions of the word vector space to encode your words for example.

*Remember the mathematical equation*

You can notice the presence of 2 unknown variables in the equation: **D** and **h**. In neural networks: 
- **h = E . x** (I avoid writing the bias terms for the sake of simplicity). With **E** a matrix

You can interpret the neural network weights as the weights of an encoder (trying to create the most sparse representation of the data it cans) and the neural network weight D as the weights of the decoder (Trying to reproduce the original content as best as it cans). Also it happens to be what we call an auto-encoder.

*Remark:*
How much neurons? you said. It seems impossible to tell without experiment, this is why we use cross validation to optimize hyper-parameters.
If the number of neurons (and so, dimensions) is greater than the intrinsic vector space of the data, then it means that you can infer an over-complete basis and so, you can achieve sparse coding in your neural network. If you don't have enough neurons, you will compress your data with some loss, getting worst results when you try to reproduce the data.

Why not using as much neurons as we can? Because this is just too expensive and terribly ineficient.

## Why having a sparse representation of data can be useful?
### Intuitions
A good sparse coding algorithm would be able to find a very good over-complete basis of vectors representing the inherent structure of the data. By "very good", I mean "meaningful" for a human:

It could be compared to:
- Decomposing all the matters in the universe in a composition of atoms
- Decomposing music in a composition of just a few notes
- Decomposing a whole language in just a few letters
And yet, from those basic components, you can generate all the music or all the matters or all the words in a language back.

Having an algorithm, which could infer those basic components just by looking at the data would be very useful, because it would imply that we could pretrain a neural network to learn *always useful features* for any future task that you would need to solve.

Also sparse code, brings a loads of interesting characteristics for the brain which can be apply to computers too and are often overlooked.

#### Interference
This scheme of neural code avoids the risk of unwanted interference between neurons: One neurons can't encode two features at the same time.

in other schemes (see dense/sparse code): one neuron might react to multiple feature, if it happens that the 2 feature appears at the same time in the stimuli, one could imagine the emergence of a new signal from a neurons reacting to both information which could be misleading.

#### Parallelism
Since not all neurons are used at the same time and interference are unlikely in sparse code, neurons can work asynchronously and parallelise the computation of information.

It is important to note that parallelism is closely linked to interference, in a dense code, since all neurons are used to encode any piece of information, trying to parallelise computation leads to a great risk of interference.

#### Fault tolerance
Having multiple neurons firing for the same stimulus can help one to recover from an error coming from a neuron. Local code is especially sensible to errors, and dense code can always recover. 

#### Memory capacity
In local code, since one neurons represent a feature you can encode only as much as neurons.

If you think about neurons as a binary entity (1 or 0), we have two state. It means that for dense code we have actually **2^N** different combination of neurons to choose from for each feature of the input data which is A LOT!

In sparse code, if we set **d** to be the average density of neurons activation per stimulus, we then have **2^d** different combination this time. It's interesting because since **d** can be a lot smaller than **N**, we have room for other characteristics.

#### Complexity And Decoding
Obviously, the more neurons you need to observe to detect what feature has been encoded, the more complex it will be to decode information.

#### Sum up
(N: total numbers of neurons, d: average density of activations, we simplify the behavior of neurons as activated or not)

|            | Local code | Sparse code | Dense code |
|------------|------------|-------------|------------|
| Interference      | None | Unlikely | Likely | 
| Parallelism       | Possible | Possible | Dangerous | 
| Decoding          | easy | Average | Complex |
| Fault tolerant    | None | Ok | Very good | 
| Memory Capacity   | N | 2^(d) | 2^N | 
| Complexity        | None | low | Very high |
| Computation power or energy        | very low | low | high | 

As you can see sparse code seems to be the perfect middle ground between having enough computation power/memory/stability and using as less as possible energy to handle the task at hand.



## How to use it in neural networks?
The simplest known usage of combining neural networks and sparse coding is in sparse auto-encoder: It is a neural network that will try to mimic the identity function while under some constraint of sparsity in the hidden layers or the objective function.

After learning the concept, I wanted to try some experiments that I've never seen around. My idea was to use the ReLU activation combined with a L1 norm in the objective functions to promote sparse neurons activations: Let's see (with code and Tensorflow) how far can we go with this idea.

### Some experiments with Tensorflow
We will now try to validate our previous statement with some very simple neural networks.
Multiple questions here:
- Can we reach (at all) local code on the MNIST dataset?
- Can we reach sparse code while keeping a good accuracy?
- Does all neurons get activated through the complete dataset?

Experiments:
- A simple fully connected neural network used as a classifier of digits on MNIST, i add the sparsity constraint with l1-norm directly on the hidden layer
    - I will test with different coefficient of sparsity, **especially 0**, given us a baseline 
- Let's move forward with a CNN, doing the same experiment as before (l1-norm on the activations of the feature map this time). Since we are trying to classify images, we should see an improvement.
- Finally, I will add a reconstruction layer and loss to our CNN architecture and train them jointly to see if any improvement happen


#### Fully connected neural network
I will do this experiment with only one hidden layer of 784 neurons because our input data is a 28x28 image so I'm sure I can reencode the data in as many dimensions as the original space. Let's see how sparse we can get!

**All code files can be found in the repo**

First we load the MNIST dataset and prepare our placeholder for training/testing:
```python
import time, os

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

dir = os.path.dirname(os.path.realpath(__file__))
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# Fully connected model
# Number of parameters: (784 * 784 + 784) + (784 * 10 + 10) = 615440 + 7850 = 623290
# Dimensionality: R^784 -> R^784 -> R^10

# Placeholder
x = tf.placeholder(tf.float32, shape=[None, 784])
y_true = tf.placeholder(tf.float32, shape=[None, 10])

sparsity_constraint = tf.placeholder(tf.float32)
```
We add our neural layer and calculate the average density of activations:
```python

# Variables
with tf.variable_scope('NeuralLayer'):
    W = tf.get_variable('W', shape=[784, 784], initializer=tf.random_normal_initializer(stddev=1e-3))
    b = tf.get_variable('b', shape=[784], initializer=tf.constant_initializer(0.0))

    z = tf.matmul(x, W) + b
    a = tf.nn.relu(z)

    # We graph the average density of neurons activation
    # We sum all activations per input data of the batch and averaged the result
    average_density = tf.reduce_mean(tf.reduce_sum(tf.cast((a > 0), tf.float32), reduction_indices=[1]))
    tf.scalar_summary('AverageDensity', average_density)
```

We finish with the softmax layer and the loss, note that we add our sparsity constraint on activations in the loss
```python
with tf.variable_scope('SoftmaxLayer'):
    W_s = tf.get_variable('W_s', shape=[784, 10], initializer=tf.random_normal_initializer(stddev=1e-3))
    b_s = tf.get_variable('b_s', shape=[10], initializer=tf.constant_initializer(0.0))

    out = tf.matmul(a, W_s) + b_s
    y = tf.nn.softmax(out)

with tf.variable_scope('Loss'):
    epsilon = 1e-7 # After some training, y can be 0 on some classes which lead to NaN 
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_true * tf.log(y + epsilon), reduction_indices=[1]))
    # We add our sparsity constraint on the activations
    # Since ReLU always output values bigger than 0, this is equivalent to lhe L1 norm
    loss = cross_entropy + sparsity_constraint * tf.reduce_sum(a)

    tf.scalar_summary('loss', loss) # Graph the loss
```
Now we are going to merge all the defined summaries and only then we will calculate accuracy and create its summary. 
We do that in this order to avoid graphing the accuracy every iteration
```python
summaries = tf.merge_all_summaries()

with tf.variable_scope('Accuracy'):
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_true, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    acc_summary = tf.scalar_summary('accuracy', accuracy) 
```
Now we finish with the training/testing part
```python
# Training
adam = tf.train.AdamOptimizer(learning_rate=1e-3)
train_op = adam.minimize(loss)
sess = None
# We iterate over different sparsity constraint
for sc in [0, 1e-4, 5e-4, 1e-3, 2.7e-3]:
    result_folder = dir + '/results/' + str(int(time.time())) + '-fc-sc' + str(sc)
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        sw = tf.train.SummaryWriter(result_folder, sess.graph)
        
        for i in range(20000):
            batch = mnist.train.next_batch(100)
            current_loss, summary, _ = sess.run([loss, summaries, train_op], feed_dict={
                x: batch[0],
                y_true: batch[1],
                sparsity_constraint: sc
            })
            sw.add_summary(summary, i + 1)

            if (i + 1) % 100 == 0:
                acc, acc_sum = sess.run([accuracy, acc_summary], feed_dict={
                    x: mnist.test.images, 
                    y_true: mnist.test.labels
                })
                sw.add_summary(acc_sum, i + 1)
                print('batch: %d, loss: %f, accuracy: %f' % (i + 1, current_loss, acc))
```

*Since we are talking about images, I've done another experiment using a convolutional neural network and 200 feature maps (Remember that the weight are shared between feature map in convolutionnal neural net, we are just looking for a local feature everywhere in the image)*

Et voila!
Let's launch our little experiment (Do it yourself to see graphs with tensorboard)
Here is the summary for (20000 iterations):

Fully connected neural net (N = 784 neurons):

![Fully connected neural net results](https://gist.githubusercontent.com/morgangiraud/9268b95bc1debd2feac37414c035ba03/raw/results-fc.jpg "Fully connected neural net results")

**Some remarks:**
- Without sparsity constraint the networks converge under a dense network (<~0.5 average activity ratio)
- We almost reach local code with an AVR(average activity ratio) of 1.67, for 89% accuracy. Clearly better than a random monkey! 
- Without loosing accuracy, we've been able to reach an AVR of 5.4 instead of 156.3 with a good sparsity parameter which is only 0.6% of activated neurons.
- It's hard to see but we reached overfitting with the sparsity constraint. The absolute best score would have been 0.981 accuracy for an AVR of 28.19 (Sparsity constraint of 1e-05)

#### Convolutionnal neural network
Convolutionnal neural net (N = 39200, note that neurons are spatially distributed):

![Convolutionnal neural net results](https://gist.githubusercontent.com/morgangiraud/9268b95bc1debd2feac37414c035ba03/raw/results-cnn.jpg "Convolutionnal neural net results")

**Some remarks:**
- We are very close to the capacity of the fully connecter neural network. 
- Without loosing accuracy, we've been able to reach an AVR of 146.2 instead of 10932 with a good sparsity parameter which is only 0.3% of activated neurons.

Finally I venture in the auto-encoder area to see if my idea makes more sense there.

#### Convolutionnal Sparse auto-encoder
Convolutionnal neural net jointly trained with a sparse auto-encoder (N = 39200):

![Autoencoder results](https://gist.githubusercontent.com/morgangiraud/9268b95bc1debd2feac37414c035ba03/raw/results-ae.jpg "Autoencoder results")

**Some remarks:**
- We actually **accelerated**(in terms of iteration, not time) and **improved** the learning of the classifier with sparsity. That's an interesting result!
- The auto-encoder got worse with sparsity, until you quantize again real values to integer, In this case, the auto-encoder didn't change much (Average squared errors per pixels < 1).

## Conclusion
It was very interesting to explore sparse coding and finally test the concept with some very simple but concrete example. Tensorflow is definitly very efficient for fast prototyping, having access to every variables and operations allows one to customize any process very easily.

I don't think this work would lead to anything interesting in terms of neural networks, but I believe it clearly shows the concept of sparse coding in this field.

Also, imposing the sparsity constraint on ReLU activations made me think about what we call: [lateral inhibition](https://en.wikipedia.org/wiki/Lateral_inhibition) in our brain. Since the sparsity constraint is applied on the global network activations, it resonate with the description found in the wikipedia article: 

*An often under-appreciated point is that although lateral inhibition is visualised in a spatial sense, it is also thought to exist in what is known as "lateral inhibition across abstract dimensions." This refers to lateral inhibition between neurons that are not adjacent in a spatial sense, but in terms of modality of stimulus. This phenomenon is thought to aid in colour discrimination*

Who knows, maybe this is an interesting journey to pursue?

**P.S.: If you find any mistakes of incoherences in this writeup, please contact me: morgan@explee.com, thank you!**

## Running the experiment using nvidia-docker
Run experiments (using GPU):

`nvidia-docker run --name sparsity --rm -v ~/gist/sparsity:/sparsity  gcr.io/tensorflow/tensorflow:0.10.0-gpu python /sparsity/sparsity.py && python /sparsity/cnn_sparsity.py && python cnn_ae_sparsity.py`

Look at tensorboard:

`nvidia-docker run --name tensorboard -it -d -p 6006:6006  -v ~/gist/sparsity:/sparsity  gcr.io/tensorflow/tensorflow:0.10.0-gpu tensorboard --logdir /sparsity/results --reload_interval 30`


## References 
- https://en.wikipedia.org/wiki/Neural_coding
- https://en.wikipedia.org/wiki/Neural_coding#Sparse_coding
- http://ufldl.stanford.edu/wiki/index.php/Sparse_Coding
- http://deeplearning.stanford.edu/wiki/index.php/Sparse_Coding:_Autoencoder_Interpretation
- http://www.scholarpedia.org/article/Sparse_coding
- http://www.scholarpedia.org/article/Donald_Olding_Hebb
- http://www.scholarpedia.org/article/Oja_learning_rule
- https://www.youtube.com/user/hugolarochelle/search?query=sparse
