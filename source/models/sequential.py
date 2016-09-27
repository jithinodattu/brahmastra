
import numpy

import theano
import theano.tensor as T 

class Sequential(object):

	def __init__(self):
		self.layers = []

	def add(self, layer):
		self.layers.add(layer)

	def train(self):
		pass

	def predict(self):
		pass

	def save(self):
		pass
