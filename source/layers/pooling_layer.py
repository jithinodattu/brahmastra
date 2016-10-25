
import numpy
import theano
import theano.tensor as T 
from theano.tensor.nnet import conv2d

from layer import Layer

class Pooling2DLayer(Layer):

	def __init__(self, poolsize):
		self.poolsize = poolsize

	def feedforward(self, input):
		pooled_out = pool.pool_2d(
			input=input,
			ds=poolsize,
			ignore_border=True
			)
		self.output = pooled_out