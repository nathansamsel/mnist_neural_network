# A Neural Network for the MNIST Dataset
![alt tag](https://github.com/nathansegan/mnist_neural_network/blob/master/scraps/number.jpg)

A neural network to classify the handwritten digits 0-9 in the MNIST dataset.

## Data

MNIST dataset stored in [data/mnist.pkl.gz](https://github.com/nathansegan/mnist_neural_network/tree/master/data).  Zipped up and stored in a pickled format.  See [mnist_loader.py](https://github.com/nathansegan/mnist_neural_network/blob/master/src/mnist_loader.py) comments to learn more.

Network weights and biases can be saved to and loaded from files in a pickled format using `save()` and ` load()` functions.  Weights and biases are saved into [weights_file](https://github.com/nathansegan/mnist_neural_network/tree/master/data) and [biases_file](https://github.com/nathansegan/mnist_neural_network/tree/master/data).

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

#### Instantiate a network.  Note: If you intend to load a network, make sure to instantiate a network of the same size as what is to be loaded.  Default shown below

```python
>>> network = neural_net.Net([784, 30, 10])
```

#### From here, you can load a saved network or you can train this 'network' from scratch. To load a previously saved network

```python
>>> network.load()
```

#### Train `network` with a single pass through the dataset using `training_data` and a learning rate 

```python
>>> network.train(training_data, 3.0)
```

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

#### To save and load 'network', simply use `save() ` and `load()` like

```python
>>> network.save()
```

```python
>>> network.load()
```

# Credits

MNIST Data and loader borrowed from [mnielsen](https://github.com/mnielsen/neural-networks-and-deep-learning)
