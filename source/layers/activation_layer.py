
from layer import Layer

class ActivationLayer(Layer):

	def __init__(self, activ_fn=None):
		super(ActivationLayer, self).__init__()
		if activ_fn:
			self.output = activ_fn(self.input)
		else:
			self.output = self.input