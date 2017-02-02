import itertools
import numpy as np
import random
import pickle
import Image
import time

np.seterr(all='ignore')

class Net(object):

	# sizes = [784, 30, 10] for mnist data
	# 784 inputs for the 28x28 images
	# 30 neuron hidden layer
	# 10 outputs to classify as 0-9
	def __init__(self, sizes):
		self.outputs = []
		self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
		self.weights = [np.random.randn(y,x) for x, y in zip(sizes[:-1], sizes[1:])]
		
	def feedforward(self, inputs):
        # calculate input to hidden layer neurons
		in_hlayer = np.dot(self.weights[0], inputs)
		
		# feed inputs of hidden layer neurons through the activation function
		out_hlayer = np.array([self.sigmoid(z) for z in in_hlayer]).reshape(30,)
		
		# keep track of outputs for use in reverse pass during the learning process
		self.outputs = out_hlayer

		# multiply the hidden layer outputs by the corresponding weights to calculate the inputs to the output layer neurons
		in_olayer = np.dot(self.weights[1], out_hlayer.transpose())

		# feed inputs of output layer neurons through the activation function
		out_olayer = np.array([self.sigmoid(z) for z in in_olayer])
		
		# return the vector of outputs from the output layer
		return out_olayer

	def train(self, training_data, test_data, learning_rate):
		# shuffle data to minimize potential for overfit (not really a problem here, but good practice)
		random.shuffle(training_data)
		
		# loop through entire set of training data
		for training_example in training_data:
			
			# grab inputs and target value from given example
			inputs, target_raw = training_example
			
			# this is flatter this target array from shape (10,1) to (10,)
			target = list(itertools.chain.from_iterable(target_raw))
			
			# run backpropagation algorithm to do the learning
			self.do_backprop(inputs, target, learning_rate)

	def do_backprop(self, inputs, target, learning_rate):
		# Forward Pass
		
		output = self.feedforward(inputs)
		
		# Reverse Pass
		
		# calculate errors of output neurons
		error_olayer = (target - output) * output * (1 - output)
		
		# change output layer weights
		for j in range(0,9):
			for i in range(0,29):
				self.weights[1][j][i] = self.weights[1][j][i] + (learning_rate * error_olayer[j] * self.outputs[i])

		# calculate hidden layer errors
		error_hlayer1 = np.zeros((30,))
		for j in range(0,29):
			for i in range(0,9):
				error_hlayer1[j] = error_hlayer1[j] + error_olayer[i] * self.weights[1][i][j]
		error_hlayer = self.outputs * (1 - self.outputs) * error_hlayer1
		
		# change hidden layer weights
		for j in range(0,29):
			for i in range(0,783):
				self.weights[0][j][i] = self.weights[0][j][i] + (3.0 * error_hlayer[j] * inputs[i])
	
	# TODO: internal error? skipped passing through the neuron? Still looks to be working to some extent...
	def imagine(self, x):
		# convert x to output vector
		output = np.zeros((10))
		output[x] = 1
		
		# multiply outputs by weights
		in_h_layer = np.zeros((30,))
		for j in range(0,29):
			for i in range(0,9):
				in_h_layer[j] = in_h_layer[j] + self.weights[1][i][j] * output[i]
		
		# multiply the reverse output of the hidden layer neurons by each weight and sum at input to recover input
		inputs = np.zeros((784))
		for i in range(0,29):
			for j in range(0,783):
				inputs[j] = inputs[j] + self.weights[0][i][j] * in_h_layer[i]
		
		# feed imagined inputs through network and compare
		best = self.feedforward(inputs)
		best_guess = np.argmax(best)
		best_value = best[best_guess]
		print "Fed the imagined {0} through the network and got a {1}".format(x, best_guess)
		left = np.delete(best, best_guess)
		second_best = left[np.argmax(left)]
		print "Second best guess was a {0}".format(np.argmax(left))
		print "Best guess was {0}% better than second best guess".format(((best_value - second_best) * 100.0) / second_best)
		
		# create image
		im = Image.new("RGB", (28, 28), "white")
		im.mode = "L"
		pix = im.load()
		size = 28, 28
		x = 0
		y = 0
		x_min = 0
		x_max = 0
		for i in inputs:
			if (i < x_min):
				x_min = i
			if i > x_max:
				x_max = i
		for i in inputs:
			j = int(((i) / (x_max - x_min)) * 255.0)
			pix[x,y] = (j, j, j)
			x += 1
			if x == 28:
				x = 0
				y += 1
		im.show()
	
	
	def sigmoid(self, z):
    	# sigmoid activation function
		return 1.0 / (1.0 + np.exp(-z))

	def evaluate(self, test_data):
		# argmax takes the best guess of the output, then we tuple it into an array with the target for each test sample
		test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
		# compare the best guess with the target and add up the right ones
		return sum(int(x == y) for (x, y) in test_results)
	
	def test(self, test_data):
		# print as a percentage
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
