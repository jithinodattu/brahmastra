
import theano
import numpy as np 
import activation_function

"""
The initial values for the weights of a hidden layer i should be uniformly 
sampled from a symmetric interval that depends on the activation function. 
For tanh activation function the interval should be 

	[ - sqrt(6.0 / (f_in + f_out)), + sqrt(6.0 / (f_in + f_out)) ], 

where fan_{in} is the number of units in the (i-1)-th layer, and fan_{out} is 
the number of units in the i-th layer. 
For the sigmoid function the interval is 

	[ - 4*sqrt(6.0 / (f_in + f_out)), + 4*sqrt(6.0 / (f_in + f_out)) ] 


reference:
	- Y. Bengio, X. Glorot, Understanding the difficulty of training 
	deep feedforward neuralnetworks, AISTATS 2010

"""

rng = np.random.RandomState()

def gloret(n_in, n_out, activ_fn):
	W_values = np.asarray(
			rng.uniform(
				low=-np.sqrt(6. / (n_in+n_out)),
				high=np.sqrt(6. / (n_in+n_out)),
				size=(n_in,n_out)
				),
			dtype=theano.config.floatX
			)
	if activ_fn == activation_function.sigmoid:
				W_values *= 4
	return W_values