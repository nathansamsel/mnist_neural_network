import mnist_loader
import neural_net

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

net = neural_net.Net([784, 30, 10])

net.load()

net.test(test_data)
