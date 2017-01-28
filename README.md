# N-Layer Neural Network using Backpropagation
![alt tag](https://github.com/nathansegan/mnist_neural_network/blob/master/scraps/number.jpg)

A neural network to classify the handwritten digits 0-9 in the MNIST dataset.

## Data

MNIST dataset stored in [data/mnist.pkl.gz](https://github.com/nathansegan/mnist_neural_network/tree/master/data).  Zipped up and stored in a pickled format.  See [mnist_loader.py](https://github.com/nathansegan/mnist_neural_network/blob/master/src/mnist_loader.py) comments to learn more.

Network weights and biases can be saved to and loaded from files in a pickled format using `save()` and ` load()` functions.  Weights and biases are saved into [data/weights_file](https://github.com/nathansegan/mnist_neural_network/tree/master/data) and [data/biases_file](https://github.com/nathansegan/mnist_neural_network/tree/master/data).

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
Note: If you intend to load a network, make sure to instantiate a network of the same size as what is to be loaded.  Default shown below

```python
>>> network = neural_net.Net([784, 30, 10])
```

Each index in the array input represents a layer in the network with the first and last representing the input and output layer respectively.  Therefore, the above example represent a three layer network with 784 inputs (28x28 images), a 30 neuron hidden layer, and an output layer (digits 0-9).  Below is another example using two hidden layers of size 20 and 30.

```python
>>> network = neural_net.Net([784, 20, 30, 10])
```

Note: To imagine inputs, the network size can only be 3 for now (input, 1 hidden layer, output).  Will be expanded soon to accomodate n-layer.


#### From here, you can load a saved network or you can train `network` from scratch. To load a previously saved network

```python
>>> network.load()
```


#### Train `network` with a single pass through the dataset using `training_data` and a learning rate 

```python
>>> network.train(training_data, 3.0)
```

Note: I have gotten up to a 92.5% success rate on `test_data` with enough training with current implementation.  Improvements coming soon.

#### Test the accuracy of `network` against `test_data` using

```python
>>> network.test(test_data)
```


#### Imagine the input of an output using `imagine()`.  This is like asking `network` the question 

> "Hey, I have trained you to be able to classify a 3.  Now, I'm curious, what do you think a 3 looks like?".

```python
>>> network.imagine(3)
```

![alt tag](https://github.com/nathansegan/mnist_neural_network/blob/master/scraps/3.png)

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

## Forward Pass
Input pattern is applied and the output is calculated

## Reverse Pass
Error of each neuron is calculated and the error is used to mathematically change the weights to minimize them, repeatedly.
_ = subscript, W+ = new weight, W = old weight, δ = error, η = learning rate.

1. Calculate errors of output neurons
```math
δ_α = out_α (1 - out_α) (Target_α - out_α)
δ_β = out_β (1 - out_β) (Target_β - out_β)
```
2. Change output layer weights
```math
W+_Aα = W_Aα + η * δα * out_A
W+_Aβ = W_Aβ + η * δβ * out_A

W+_Bα = W_Bα + η * δα * out_B
W+_Bβ = W_Bβ + η * δβ * out_B

W+_Cα = W_Cα + η * δα * out_C
W+_Cβ = W_Cβ + η * δβ * out_C
```

3. Calculate (back-propagate) hidden layer errors
```math
δ_A = out_A * (1 – out_A) * (δ_α * W_Aα + δ_β * W_Aβ)
δ_B = out_B * (1 – out_B) * (δ_α * W_Bα + δ_β * W_Bβ)
δ_C = out_C * (1 – out_C) * (δ_α * W_Cα + δ_β * W_Cβ)
```

4. Change hidden layer weights
```math
W+_λA = W_λA + η * δ_A * in_λ 
W+_ΩA = W_ΩA + η * δ_A * in_Ω

W+_λB = W_λB + η * δ_B * in_λ 
W+_ΩB = W_ΩB + η * δ_B * in_Ω

W+_λC = W_λC + η * δ_C * in_λ
W+_ΩC = W_ΩC + η * δ_C * in_Ω
```
# Credits

MNIST Data and loader borrowed from [mnielsen](https://github.com/mnielsen/neural-networks-and-deep-learning)
Network image and math for backpropagation from [here](https://www.fer.unizg.hr/_download/repository/BP_chapter3_-_bp.pdf)
