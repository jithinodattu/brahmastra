
from theano import tensor as T

def sigmoid(z):
	return 1.0/(1.0 + T.exp(-z))