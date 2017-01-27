# mnist_neural_network
A neural network to classify the handwritten digits 0-9 for the MNIST dataset.

## Data
MNIST Data and loader borrowed from [mnielsen](https://github.com/mnielsen/neural-networks-and-deep-learning)

MNIST dataset stored in [data/mnist.pkl.gz](https://github.com/nathansegan/mnist_neural_network/tree/master/data).  Zipped in a pickled format.  See [mnist_loader.py](https://github.com/nathansegan/mnist_neural_network/blob/master/src/mnist_loader.py) comments to learn more.

Network weights and biases can be saved to and loaded from files in a pickled format using `save()` and ` load()` functions.  Weights and biases are saved into [weights_file](https://github.com/nathansegan/mnist_neural_network/tree/master/data) and [biases_file](https://github.com/nathansegan/mnist_neural_network/tree/master/data).

## Code

### To test currently saved network, run:

```shell
python test_net.py
```

### To work with network more interactively, run

```shell
python
```

Import needed modules

```shell
>>> import neural_net
```
```shell
>>> import mnist_loader
```
