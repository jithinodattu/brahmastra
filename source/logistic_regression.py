

from layers import *
from models import *
from utils import load_data

mnist_data = load_data.mnist()

model = Sequential()
model.add(InputLayer(input=mnist_data, n_in=28 * 28, n_out=10))
model.add(SoftMaxLayer(n_in=28 * 28, n_out=10))
model.add(OutputLayer())

model.optimize()