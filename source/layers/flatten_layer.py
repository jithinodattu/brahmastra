
import numpy
import theano
import theano.tensor as T 
from theano.tensor.nnet import conv2d

from layer import Layer

class FlattenLayer(Layer):

	def __init__(self, order=2):
		self.order = order

	def feedforward(self, input):
		self.output = input.flatten(self.order)