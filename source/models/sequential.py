
import numpy

import theano
import theano.tensor as T 

class Sequential(object):

	def __init__(self):
		self.layers = []

	def add(self, layer):
		self.layers.append(layer)

	def _connect_layers(self):
		for layer, next_layer in zip(self.layers[:-2], self.layers[1:-1]):
			next_layer.input = layer.output
		self.input = self.layers[0].output
		self.p_y_given_x = self.layers[-2].output
		self.output = self.layers[-1].output

	def negative_log_likelihood(self, y):
		return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

	def errors(self, y):
		if y.ndim != self.y_pred.ndim:
			raise TypeError(
				'y should have the same shape as self.y_pred',
				('y', y.type, 'y_pred', self.y_pred.type)
			)
		if y.dtype.startswith('int'):
			return T.mean(T.neq(self.output, y))
		else:
			raise NotImplementedError()

	def train(learning_rate=0.13, n_epochs=1000, datasets, batch_size=600):
		train_set_x, train_set_y = datasets[0]
		valid_set_x, valid_set_y = datasets[1]
		test_set_x, test_set_y = datasets[2]

		n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
		n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
		n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size

		index = T.lscalar()
		x = T.matrix('x')
		y = T.ivector('y')

		cost = classifier.negative_log_likelihood(y)

		test_model = theano.function(
			inputs=[index],
			outputs=self.errors(y),
			givens={
				x: test_set_x[index * batch_size: (index + 1) * batch_size],
				y: test_set_y[index * batch_size: (index + 1) * batch_size]
			}
		)

		validate_model = theano.function(
			inputs=[index],
			outputs=self.errors(y),
			givens={
				x: valid_set_x[index * batch_size: (index + 1) * batch_size],
				y: valid_set_y[index * batch_size: (index + 1) * batch_size]
			}
		)

		g_W = T.grad(cost=cost, wrt=self.layers[1].W)
		g_b = T.grad(cost=cost, wrt=self.layers[1].b)

		updates = [(self.layers[1].W, self.layers[1].W - learning_rate * g_W),
			   (self.layers[1].b, self.layers[1].b - learning_rate * g_b)]

		train_model = theano.function(
			inputs=[index],
			outputs=cost,
			updates=updates,
			givens={
				x: train_set_x[index * batch_size: (index + 1) * batch_size],
				y: train_set_y[index * batch_size: (index + 1) * batch_size]
			}
		)

		patience = 5000
		patience_increase = 2
		improvement_threshold = 0.995
		validation_frequency = min(n_train_batches, patience // 2)

		best_validation_loss = numpy.inf
		test_score = 0.
		start_time = timeit.default_timer()

		done_looping = False

		epoch = 0
		while (epoch < n_epochs) and (not done_looping):
			epoch = epoch + 1
			for minibatch_index in range(n_train_batches):

				minibatch_avg_cost = train_model(minibatch_index)
				iter = (epoch - 1) * n_train_batches + minibatch_index

				if (iter + 1) % validation_frequency == 0:
					validation_losses = [validate_model(i)
										 for i in range(n_valid_batches)]
					this_validation_loss = numpy.mean(validation_losses)

					print(
						'epoch %i, minibatch %i/%i, validation error %f %%' %
						(
							epoch,
							minibatch_index + 1,
							n_train_batches,
							this_validation_loss * 100.
						)
					)

					if this_validation_loss < best_validation_loss:
						if this_validation_loss < best_validation_loss *  \
						   improvement_threshold:
							patience = max(patience, iter * patience_increase)

						best_validation_loss = this_validation_loss

						test_losses = [test_model(i)
									   for i in range(n_test_batches)]
						test_score = numpy.mean(test_losses)

						print(
							(
								'	 epoch %i, minibatch %i/%i, test error of'
								' best model %f %%'
							) %
							(
								epoch,
								minibatch_index + 1,
								n_train_batches,
								test_score * 100.
							)
						)

						with open('best_model.pkl', 'wb') as f:
							pickle.dump(classifier, f)

				if patience <= iter:
					done_looping = True
					break

		end_time = timeit.default_timer()
		print(
			(
				'Optimization complete with best validation score of %f %%,'
				'with test performance %f %%'
			)
			% (best_validation_loss * 100., test_score * 100.)
		)
		print('The code run for %d epochs, with %f epochs/sec' % (
			epoch, 1. * epoch / (end_time - start_time)))
		print(('The code for file ' +
			   os.path.split(__file__)[1] +
			   ' ran for %.1fs' % ((end_time - start_time))), file=sys.stderr)

	def predict(self):
		pass

	def save(self):
		pass
