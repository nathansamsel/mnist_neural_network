# Neural Network using Backpropagation
![alt tag](https://github.com/nathansegan/mnist_neural_network/blob/master/scraps/number.jpg)

A neural network to classify the handwritten digits 0-9 in the MNIST dataset.  This network uses 784 inputs (for the 28x28 images of the handwritten digits), 30 neuron hidden layer, and 10 outputs.

## Data

MNIST dataset stored in [data/mnist.pkl.gz](https://github.com/nathansegan/mnist_neural_network/tree/master/data).  Zipped up and stored in a pickled format.  See [mnist_loader.py](https://github.com/nathansegan/mnist_neural_network/blob/master/src/mnist_loader.py) comments to learn more.

Network weights can be saved to and loaded from a file in a pickled format using `save()` and ` load()` functions.  Weights are saved into [data/weights_file](https://github.com/nathansegan/mnist_neural_network/tree/master/data).

## Dependencies

All code was developed using **Python 2.7.12** on an Ubuntu 16.04 LTS system.

Libraries
* Numpy
* Image
* pickle
* random
* gzip

## Code

### To test currently saved network, run

```shell
$ python test_net.py
```


### To work with network more interactively, run

```shell
$ python
```


#### Import needed modules

```python
>>> import neural_net
```
```python
>>> import mnist_loader
```


#### Load data

```python
>>> training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
```


#### Instantiate a network 

```python
>>> network = neural_net.Net([784, 30, 10])
```

Each index in the array input represents a layer in the network with the first and last representing the input and output layer respectively.  Therefore, the above example represent a three layer network with 784 inputs (28x28 images), a 30 neuron hidden layer, and an output layer (digits 0-9).  For now, this is the only network design that works because the layers are hardcoded in, but it will soon be abstracted to an n-layer network where you can use a network design like below

```python
>>> network = neural_net.Net([784, 20, 30, 10])
```

#### From here, you can load a saved network or you can train `network` from scratch. To load a previously saved network

```python
>>> network.load()
```

#### Train `network` with a single pass through the dataset using `training_data` and a learning rate 

```python
>>> network.train(training_data, 3.0)
```

Note: Training takes O(n^2) time now (a really long time for the whole data set) because I wanted to be very explicit about how errors and weights were being calculated.  I did this so that it follows the backpropagation process explained below very closely and clearly.  This makes it easy for people to see exactly what happening without getting confused by any vector/matrix math.  I may create a new project for a more efficient version, because I like this as proof of concept and learning tool.

#### Test the accuracy of `network` against `test_data` using

```python
>>> network.test(test_data)
```


#### Imagine the input of an output using `imagine()`

This is like asking `network` the question 
> "Hey, I have trained you to be able to classify a 3.  Now, I'm curious, what do you think a 3 looks like?".

```python
>>> network.imagine(3)
```

![alt tag](https://github.com/nathansegan/mnist_neural_network/blob/master/scraps/number_3.png)

Think of the output like a heatmap that indicates what pixels are most important in differentiating a given digit from any other digit.


#### To save and load `network`, simply use `save() ` and `load()` like

```python
>>> network.save()
```

```python
>>> network.load()
```

# Backpropogation Process

![alt tag](https://github.com/nathansegan/mnist_neural_network/blob/master/scraps/sample_network.png)

Using the sigmoid activation function:
```math
1 / (1 + e(-z))
```

## Forward Pass
Input pattern is applied and the output is calculated


#### Calculate the input to the hidden layer neurons
```math
in_A = W_ΩA * Ω + W_λA * λ
in_B = W_ΩB * Ω + W_λB * λ
in_C = W_ΩC * Ω + W_λC * λ
```

#### Feed inputs of hidden layer neurons through the activation function
```math
out_A = 1 / (1 + e^( -1 * in_A))
out_B = 1 / (1 + e^( -1 * in_B))
out_C = 1 / (1 + e^( -1 * in_C))
```

#### Multiply the hidden layer outputs by the corresponding weights to calculate the inputs to the output layer neurons
```math
in_α = out_A * W_Aα + out_B * W_Bα + out_C * W_Cα
in_β = out_A * W_Aβ + out_B * W_Bβ + out_C * W_Cβ
```

#### Feed inputs of output layer neurons through the activation function
```math
out_α = 1 / (1 + e^( -1 * in_α))
out_β = 1 / (1 + e^( -1 * in_β))
```


## Reverse Pass
Error of each neuron is calculated and the error is used to mathematically change the weights to minimize them, repeatedly.

_ = subscript, W+ = new weight, W = old weight, δ = error, η = learning rate.

#### Calculate errors of output neurons
```math
δ_α = out_α * (1 - out_α) * (Target_α - out_α)
δ_β = out_β * (1 - out_β) * (Target_β - out_β)
```

#### Change output layer weights
```math
W+_Aα = W_Aα + η * δα * out_A
W+_Aβ = W_Aβ + η * δβ * out_A

W+_Bα = W_Bα + η * δα * out_B
W+_Bβ = W_Bβ + η * δβ * out_B

W+_Cα = W_Cα + η * δα * out_C
W+_Cβ = W_Cβ + η * δβ * out_C
```

#### Calculate (back-propagate) hidden layer errors
```math
δ_A = out_A * (1 – out_A) * (δ_α * W_Aα + δ_β * W_Aβ)
δ_B = out_B * (1 – out_B) * (δ_α * W_Bα + δ_β * W_Bβ)
δ_C = out_C * (1 – out_C) * (δ_α * W_Cα + δ_β * W_Cβ)
```

#### Change hidden layer weights
```math
W+_λA = W_λA + η * δ_A * in_λ 
W+_ΩA = W_ΩA + η * δ_A * in_Ω

W+_λB = W_λB + η * δ_B * in_λ 
W+_ΩB = W_ΩB + η * δ_B * in_Ω

W+_λC = W_λC + η * δ_C * in_λ
W+_ΩC = W_ΩC + η * δ_C * in_Ω
```

## Solving for inputs
> "Hey, I have trained you to be able to classify a 3.  Now, I'm curious, what do you think a 3 looks like?".

Note: Assuming a trained network and desired outputs provided.  Output provided as a vector of 0s and 1s.  For example: [0,0,0,1,0,0,0,0,0,0] is the output for a 3.  Below math is for network example above, but can be applied more generally.

Note: This math is not entirely correct but seems to be giving some reasonable results.  Also, `in_A, in_B, in_C` refers to the reverse input into a because this whole process is taking the desired output and passing it backwards to solve for inputs.

#### Multiply the desired output by the output layer weights
```math
in_A = out_α * W_Aα + out_β * W_Aβ
in_B = out_α * W_Bα + out_β * W_Bβ
in_C = out_α * W_Cα + out_β * W_Cβ
```

#### Multiply `in_A, in_B, in_C` by each hidden layer weight
```math
Ω = in_A * W_ΩA + in_B * W_ΩB + in_C * W_ΩC
λ = in_A * W_λA + in_B * W_λB + in_C * W_λC
```

Now, to test your imagined input, feed the input back through the network and see if you classify it correctly!

Note: Don't forget to normalize inputs once you've solved for them, or else it will just look like noise!

# Credits

MNIST Data and loader borrowed from [mnielsen](https://github.com/mnielsen/neural-networks-and-deep-learning)

Network image and math for backpropagation from [here](https://www.fer.unizg.hr/_download/repository/BP_chapter3_-_bp.pdf)
