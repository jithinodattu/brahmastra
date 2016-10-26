
import numpy
import theano
import theano.tensor as T 

from layer import Layer

class DenseLayer(Layer):

	def __init__(self, n_in, n_out, activ_fn=None, w_initializer=None):
		super(DenseLayer, self).__init__()

		if w_initializer:
			W_value = w_initializer(n_in, n_out, (n_in,n_out), activ_fn)
		else:
			W_value = numpy.zeros(
				(n_in, n_out),
				dtype=theano.config.floatX
				)


		self.W = theano.shared(
			value=W_value,
			name='W',
			borrow=True
			)

		b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
		self.b = theano.shared(
			value=b_values,
			name='b',
			borrow=True
			)

		self.activ_fn = activ_fn

		self.params = [self.W, self.b]

	def feedforward(self, input):
		if self.activ_fn:
			self.output = self.activ_fn(T.dot(input, self.W) + self.b)
		else:
			self.output = T.dot(input, self.W) + self.b