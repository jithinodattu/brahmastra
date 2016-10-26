
import numpy
import theano
import theano.tensor as T 
from theano.tensor.signal import pool

from layer import Layer

class Pooling2DLayer(Layer):

	def __init__(self, poolsize):
		self.poolsize = poolsize

	def feedforward(self, input):
		pooled_out = pool.pool_2d(
			input=input,
			ds=self.poolsize,
			ignore_border=True
			)
		self.output = pooled_out