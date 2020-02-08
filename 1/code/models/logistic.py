import numpy as np
# from ..myutility import convertToOneHot

class logistic_regression():
	def __init__(self, num_of_features, num_of_class, max_iter=100):
		assert isinstance(num_of_features, int) and num_of_features > 0
		assert isinstance(max_iter, int) and max_iter > 0
		assert isinstance(num_of_class, int) and num_of_class > 0

		self.weight = 2 * np.random.random_sample( (num_of_features, num_of_class) ) - 1
		# self.weight = 2 * np.random.random_sample( (num_of_class, num_of_features) ) - 1
		self.num_of_features = num_of_features
		self.max_iter = max_iter
		self.num_of_class = num_of_class

	def fit(self, X, y, lr=0.001):
		assert type(X) is np.ndarray and type(y) is np.ndarray and (X.shape[0] == y.shape[0])
		assert self.num_of_features == X.shape[1]
		assert self.num_of_class == y.shape[1]
		num_of_samples = X.shape[0]
		
		predict_y = np.zeros((num_of_samples, self.num_of_class))
		# print(self.weight.shape)
		# print(X.shape)
		for i in range(self.max_iter):
			# predict_y = np.exp(-np.matmul(X, self.weight)) / (1 + np.sum(np.exp(-np.matmul(X, self.weight))))
			predict_y = 1 / (1 + np.exp( - np.matmul(X, self.weight)))
			gradient = np.dot(X.T, predict_y - y)
			# print(gradient.shape)
			# if gradient.ndim == 2:
			# 	gradient = np.sum(gradient, axis=1)
			self.weight -= gradient * lr
		# print(self.weight.shape)
		return self

	def predict(self, X):
		# predict_y = np.exp(-np.matmul(X, self.weight)) / (1 + np.sum(np.exp(-np.matmul(X, self.weight))))
		predict_y = 1 / (1 + np.exp( - np.matmul(X, self.weight)))
		# y = np.where(predict_y < 0.5, 0, 1) 
		# max_y = np.max(predict_y, axis=1)
		# y = np.where(predict_y == max_y)
		# print(predict_y)
		# print(y)
		y = np.argmax(predict_y, axis=1)
		return y


# train_sample = 120
# test_sample = 30
# num_of_features = 4

# train_X, train_y = np.ones((train_sample, num_of_features)), np.zeros(train_sample)
# test_X, test_y = np.ones((test_sample, num_of_features)), np.zeros(test_sample)
# LR = logistic_regression(num_of_features)
# LR.fit(train_X, train_y)
# predict_y = LR.predict(test_X)
# print(predict_y)

