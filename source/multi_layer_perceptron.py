"""
Single neuron can not perform well when the data points plotted on the n dimensional 
feature space are not seperable by a hyperplane. Minsky and Papert has proved this 
in their book `Perceptrons`. Classic example for single layer neural nets showing 
poor performance is XOR problem. Stacking a layers above the single input layer 
can solve this problem where neurons in the first layers does some non linear 
tranformation and represents the input into another higher dimensions which is 
linearly seperable by the output layer. K. Hornik showed that neural net with 
single hidden layer can be universal approximators provided sufficient number 
of neurons are present in the hidden layers
"""

from layers import *
from models import *
from utils import load_data
from utils.activation_function import tanh
from utils.weight_initializer import gloret
from utils.regularizer import L2_sqr

mnist_data = load_data.mnist()

input_dim = 28 * 28
hidden_dim = 500
output_dim = 10

model = Sequential()

"""
Hidden layer: Here the network performs non linear tranformation (here we do tanh) ,
and represents the input into some higher dimentions (here we set hidden_dim=500), 
"""
model.add(
	DenseLayer(
		n_in=input_dim, 
		n_out=hidden_dim,
		activ_fn=tanh,
		w_initializer=gloret
		))
"""
Output of the hidden layer is fed to outputlayer which will act a logistic regression 
classifier.
"""
model.add(
	DenseLayer(
		n_in=hidden_dim, 
		n_out=output_dim,
		activ_fn=tanh,
		w_initializer=gloret
		))

model.add(SoftMaxLayer())

model.optimize(
	dataset=mnist_data,
	regularizer=L2_sqr,
	reg_lambda=0.0001
	)