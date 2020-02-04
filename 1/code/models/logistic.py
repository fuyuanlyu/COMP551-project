import numpy as np

class logistic_regression():
	def __init__(self, num_of_features, max_iter=1000):
		assert isinstance(num_of_features, int)
		assert num_of_features > 0
		assert isinstance(max_iter, int)
		assert max_iter > 0

		self.weight = np.zeros(num_of_features)
		self.num_of_features = num_of_features
		self.max_iter = max_iter
		self.cost = 0.

	def fit(self, X, y, lr=0.001):
		assert (X.shape[0] == y.shape[0])
		num_of_samples, num_of_features = X.shape[0], X.shape[1]
		assert self.num_of_features == num_of_features
		for i in range(self.max_iter):
			z = np.dot(X, self.weight)
			self.cost = np.mean( y * np.log1p(np.exp(-z)) + (1-y) * np.log1p(np.exp(z)) )
			predict_y = 1 / (1 + np.exp( - np.dot(X, self.weight.T)))
			gradient = np.dot(X.T, predict_y - y)
			self.weight -= gradient * lr
		return self

	def predict(self, X):
		predict_y = 1 / (1 + np.exp( - np.dot(X, self.weight.T)))
		y = np.where(predict_y < 0.5, 0, 1) 
		return y


train_sample = 20
test_sample = 7
num_of_features = 8

train_X, train_y = np.ones((train_sample, num_of_features)), np.zeros(train_sample)
test_X, test_y = np.ones((test_sample, num_of_features)), np.zeros(test_sample)
LR = logistic_regression(num_of_features)
LR.fit(train_X, train_y)
predict_y = LR.predict(test_X)
print(predict_y)

