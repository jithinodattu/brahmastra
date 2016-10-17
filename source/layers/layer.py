
import theano
import theano.tensor as T 

class Layer(object):
	def __init__(self):
		self.input = T.matrix('input')
		self.output = T.ivector('output')