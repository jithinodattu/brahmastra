
from layer import Layer

class ActivationLayer(Layer):

	def __init__(self, activ_fn=None):
		super(ActivationLayer, self).__init__()
		self.activ_fn=activ_fn

	def feedforward(self, input):
		if self.activ_fn:
			self.output = self.activ_fn(input)
		else:
			self.output = input