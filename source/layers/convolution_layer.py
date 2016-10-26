
import numpy
import theano
import theano.tensor as T 
from theano.tensor.nnet import conv2d

from layer import Layer
from utils.weight_initializer import gloret

class Convolution2DLayer(Layer):

	def __init__(self, filter_shape, image_shape, activ_fn=None, w_initializer=gloret):

		n_in = numpy.prod(filter_shape[1:])
		n_out = filter_shape[0]
		if w_initializer:
			W_value = w_initializer(n_in, n_out, filter_shape, activ_fn)
		else:
			W_value = numpy.zeros(
				filter_shape,
				dtype=theano.config.floatX
				)

		self.W = theano.shared(
			value=W_value,
			name='W',
			borrow=True
			)

		b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
		self.b = theano.shared(value=b_values, borrow=True)

		self.filter_shape = filter_shape
		self.image_shape = image_shape
		self.activ_fn = activ_fn

		self.params = [self.W, self.b]

	def feedforward(self, input):
		input = input.reshape(self.image_shape)
		conv_out = conv2d(
			input=input,
			filters=self.W,
			filter_shape=self.filter_shape,
			input_shape=self.image_shape
			)

		output = conv_out + self.b.dimshuffle('x', 0, 'x', 'x')
		if self.activ_fn:
			self.output =self.activ_fn(output)
		else:
			self.output = output