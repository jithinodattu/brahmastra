

from layers import *
from models import *
from utils import load_data

mnist_data = load_data.mnist()

input_dim = 28 * 28
output_dim = 10

model = Sequential()
model.add(DenseLayer(n_in=input_dim, n_out=output_dim))
model.add(ActivationLayer(activ_fn=None))
model.add(SoftMaxLayer())

model.optimize(dataset=mnist_data)