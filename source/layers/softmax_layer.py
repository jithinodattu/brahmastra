
import numpy as np

import theano
import theano.tensor as T 

from layer import Layer

class SoftMaxLayer(Layer):

	def __init__(self):
		super(SoftMaxLayer, self).__init__()
		
	def feedforward(self, input):
		self.output = T.nnet.softmax(input)