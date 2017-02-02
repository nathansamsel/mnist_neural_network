import mnist_loader
import neural_net

# load dataset
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

# create subset of trainng data
training_data_subset = []
for i in range(0,10000):
	training_data_subset.append(training_data[i])
	
# neural_net.Net will only accept [784, 30, 10] -- will not work otherwise
net = neural_net.Net([784, 30, 10])

# subset used to save time because it takes forever O(n^2) to train
# pass in subset of training data and learning rate
#net.train(training_data_subset, .3)

# load saved network
net.load()

# solve for inputs
net.imagine(3)

# test the trained network
#net.test(test_data)

# save weights
net.save()
