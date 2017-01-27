from PIL import Image
import os, sys
import mnist_loader
import neural_net
import random
import numpy as np
import math

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
net = neural_net.Net([784, 30, 10])
net.load()

output = [0,0,0,0,0,0,0,0,0,1]
h_layer = np.dot(output, net.weights[1])
# hidden layer inverse sigmoid = learning_rate * (activations - target) * (sigmoid(z)) * (1 - sigmoid(z))
h_layer_inv_sig = np.array([3.0 * net.cost_derivative(i, output) * net.sigmoid_derivative(i) for i in h_layer])
input_layer = np.dot(h_layer_inv_sig.transpose(), net.weights[0])

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
