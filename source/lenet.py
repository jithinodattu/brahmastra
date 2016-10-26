
from layers import *
from models import *
from utils import load_data
from utils.activation_function import relu
from utils.weight_initializer import gloret

mnist_data = load_data.mnist()

input_dim = 28 * 28
hidden_dim = 500
output_dim = 10
nkerns=[20, 50]
batch_size=500

model = Sequential()

model.add(
	Convolution2DLayer(
		filter_shape=(nkerns[0], 1, 5, 5),
		image_shape=(batch_size, 1, 28, 28),
		activ_fn=relu
		)
	)

model.add(
	Pooling2DLayer(
		poolsize=(2,2)
		)
	)

model.add(
	Convolution2DLayer(
		filter_shape=(nkerns[1], nkerns[0], 5, 5),
		image_shape=(batch_size, nkerns[0], 12, 12),
		activ_fn=relu))

model.add(
	Pooling2DLayer(
		poolsize=(2,2)
		)
	)

model.add(FlattenLayer())

model.add(
	DenseLayer(
		n_in=nkerns[1] * 4 * 4, 
		n_out=500,
		activ_fn=relu,
		w_initializer=gloret
		)
	)

model.add(
	DenseLayer(
		n_in=500, 
		n_out=10,
		activ_fn=None,
		w_initializer=gloret
		)
	)

model.add(SoftMaxLayer())

model.optimize(
	dataset=mnist_data,
	batch_size=batch_size
	)