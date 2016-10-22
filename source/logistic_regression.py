"""
Logistic Regression is a probabilistic linear classifier. It gives a probabilistic 
distribution of output conditioned over input and model parameters.
argmax function returns the class where the input belongs to, as it finds the highest 
probability among the possible classes.

maths:
	P(Y=i|X,W,b) = softmax( W*X + b )
	y_ = argmax( P(Y=i|X,W,b) )

references:
	1. "Pattern Recognition and Machine Learning" by Christopher M. Bishop
"""

from layers import *
from models import *
from utils import load_data

"""
Dataset should include training, validation and testing sets.
load_data utility functions return a tuple containing these three sets of data.
mnist_data contains normalized input data and its labels, partitioned as 
60% for training 20% for testing and remaining 20% for validation purposes.
"""
mnist_data = load_data.mnist()

"""
Each image in MNIST dataset is a 28x28 sized grey scaled image. Task is to map 
the normalized 2-D tensor of image pixels to its corresponding class, 
which are [ 0 - 9 ] numbers.
"""
input_dim = 28 * 28
output_dim = 10

"""
Logistic regression does not involve non-linear tranformation of inputs. Hence 
activation function is set to None.
"""
model = Sequential()

model.add(
	DenseLayer(
		n_in=input_dim, 
		n_out=output_dim,
		activ_fn=None
		)
	)

model.add(SoftMaxLayer())

model.optimize(dataset=mnist_data)