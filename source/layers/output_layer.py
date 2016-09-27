
import theano
import theano.tensor as T 


class OutputLayer(object):

	def __init__(self, input):

		self.input = input
		self.output = T.argmax(input, axis=1)