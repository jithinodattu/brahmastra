
import numpy as np

import theano
import theano.tensor as T 

class SoftMaxLayer(object):

	def __init__(self, n_in, n_out):

		self.W = theano.shared(
			value=np.zeros(
				(n_in, n_out),
				dtype=theano.config.floatX),
			name='W',
			borrow=True
			)

		self.b = theano.shared(
			value=np.zeros(
				(n_out,),
				dtype=theano.config.floatX),
			name='b',
			borrow=True
			)

		self.input = None

		self.output = T.nnet.softmax(T.dot(self.input, self.W) + self.b)

		self.params = [self.W, self.b]