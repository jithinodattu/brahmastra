
from layers import *
from models import *
from utils import load_data
from utils.activation_function import sigmoid

mnist_data = load_data.mnist()

input_dim = 28 * 28
hidden_dim = 500
output_dim = 10

model = Sequential()

model.add(DenseLayer(n_in=input_dim, n_out=hidden_dim))
model.add(ActivationLayer(activ_fn=sigmoid))

model.add(DenseLayer(n_in=hidden_dim, n_out=output_dim))
model.add(ActivationLayer(activ_fn=sigmoid))

model.add(SoftMaxLayer())

model.optimize(dataset=mnist_data)