
import numpy
import theano
import theano.tensor as T 

from layer import Layer

class DenseLayer(Layer):

	def __init__(self, n_in, n_out):
		super(DenseLayer, self).__init__()

		self.W = theano.shared(
			value=numpy.zeros(
				(n_in, n_out),
				dtype=theano.config.floatX
				),
			name='W',
			borrow=True
			)

		self.b = theano.shared(
			value=numpy.zeros(
				(n_out,),
				dtype=theano.config.floatX
				),
			name='b',
			borrow=True
			)

		self.output = T.dot(self.input, self.W) + self.b