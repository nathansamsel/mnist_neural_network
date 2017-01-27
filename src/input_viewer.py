from PIL import Image
import random
import numpy as np
import os, sys


def Input_Viewer(object):
	
	def __init__(self, pixel_data):
		self.pixel_data = pixel_data
		
	def create(self):
		print "{0}".format(self.pixel_data)
		im = Image.new("RGB", (28, 28), "white")
		im.mode = "1"
		pix = im.load()
		size = 28, 28
		x = 0
		y = 0
		for i in self.pixel_data:
			i = int(i * 255)
			pix[x,y] = (i, i, i)
			x += 1
			if x == 28:
				x = 0
				y += 1
		
		im.show()	
