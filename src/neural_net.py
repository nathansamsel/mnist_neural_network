import random
import numpy as np
import pickle
from PIL import Image

np.seterr(all='ignore')

class Net(object):

	def __init__(self, sizes):
		self.sizes = sizes
		self.num_layers = len(sizes)
		self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
		self.weights = [np.random.randn(y,x) for x, y in zip(sizes[:-1], sizes[1:])]
		self.imagined_inputs = [np.zeros((784))]
		
	def feedforward(self, a):
        # Return the output of the network if 'a' is input
        # where a is the vector of activations
		for b, w in zip(self.biases, self.weights):
			a = self.sigmoid(np.dot(w, a) + b)
		return a

	def train(self, training_data, learning_rate):
		# shuffle data to minimize potential for overfit
		random.shuffle(training_data)
		
		# loop through entire set of training data
		for training_example in training_data:
			
			# do backprop on training example
			(image, target) = training_example
			delta_bias_gradient, delta_weight_gradient = self.backprop(image, target)
			
			# update weights
			self.weights = [w - learning_rate * nw for w, nw in zip(self.weights, delta_weight_gradient)]
			self.biases = [b - learning_rate * nb for b, nb in zip(self.biases, delta_bias_gradient)]
			self.imagined_inputs = [i - learning_rate * nw for i, nw in zip(self.imagined_inputs, delta_weight_gradient)]

	def backprop(self, image, target):
		bias_gradient = [np.zeros(b.shape) for b in self.biases]
		weight_gradient = [np.zeros(w.shape) for w in self.weights]
		
		# feed forward
		activation = image
		activations = [image] # list to store activations layer by layer
		zs = [] # list to store z vectors, layer by layer
		for b, w in zip(self.biases, self.weights):
			z = np.dot(w, activation)+b
			zs.append(z)
			activation = self.sigmoid(z)
			activations.append(activation)
		
		#backward pass
		delta = self.cost_derivative(activations[-1], target) * self.sigmoid_derivative(zs[-1])
		bias_gradient[-1] = delta
		weight_gradient[-1] = np.dot(delta, activations[-2].transpose())
		
		for l in xrange(2, self.num_layers):
			z = zs[-l]
			delta = np.dot(self.weights[-l+1].transpose(), delta) * self.sigmoid_derivative(z)
			bias_gradient[-l] = delta
			weight_gradient[-l] = np.dot(delta, activations[-l-1].transpose())
		return (bias_gradient, weight_gradient)
	
	def imagine(self, x):
		output = np.zeros((10))
		output[x] = 1
		h_layer = np.dot(output, self.weights[1])
		# hidden layer inverse sigmoid = learning_rate * (activations - target) * (sigmoid(z)) * (1 - sigmoid(z))
		h_layer_inv_sig = np.array([3.0 * self.cost_derivative(i, output) * self.sigmoid_derivative(i) for i in h_layer])
		input_layer = np.dot(h_layer_inv_sig.transpose(), self.weights[0])
		im = Image.new("RGB", (28, 28), "white")
		im.mode = "L"
		pix = im.load()
		size = 28, 28
		x = 0
		y = 0
		x_min = 0
		x_max = 0
		for i in input_layer[0]:
			if (i < x_min):
				x_min = i
			if i > x_max:
				x_max = i
		for i in input_layer[0]:
			j = int(((i) / (x_max - x_min)) * 255.0)
			pix[x,y] = (j, j, j)
			x += 1
			if x == 28:
				x = 0
				y += 1
		im.show()
	
	def cost_derivative(self, output_activations, target):
		# return vector of partial derivatives of cost function relative to output
		return (output_activations - target)
	
	def sigmoid(self, z):
    	# sigmoid activation function
		return 1.0 / (1.0 + np.exp(-z))

	def sigmoid_derivative(self, z):
    	# sigmoid derivative
		return self.sigmoid(z) * (1 - self.sigmoid(z))

	def evaluate(self, test_data):
		test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
		return sum(int(x == y) for (x, y) in test_results)
		
	def test(self, test_data):
		print "{0}% correct!".format((float(self.evaluate(test_data)) / float(len(test_data))) * 100.0)
		
	def save(self):
		with open('../data/weights_file', 'wb') as wf:
			pickle.dump(self.weights, wf)
		with open('../data/biases_file', 'wb') as bf:
			pickle.dump(self.biases, bf)
		
	def load(self):
		with open('../data/weights_file', 'rb') as wf:
			self.weights = pickle.load(wf)
		with open('../data/biases_file', 'rb') as bf:
			self.biases = pickle.load(bf)
