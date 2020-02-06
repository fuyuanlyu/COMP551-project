import numpy as np
# from ..myutility import convertToOneHot

class BernoulliNB():
	def fit(self, X, y):
		a = X

	def predict(self, X):
		y = 0
		return y


class CategoricalNB():
	def __init__(self, num_of_features, num_of_class, alpha = 1.0):
		assert isinstance(num_of_features, int)
		assert num_of_features > 0
		assert isinstance(num_of_class, int)
		assert num_of_class > 0
		assert alpha >= 0

		self.num_of_features = num_of_features
		self.num_of_class = num_of_class
		self.alpha = alpha
		self.prior = np.zeros(self.num_of_class)
		self.likelihood = np.zeros((self.num_of_features, self.num_of_class))
		self.evidence = np.ones(self.num_of_features)

	def fit(self, X, y):
		a = X

	def predict(self, X):
		# print(X.shape)
		# print(self.weight.shape)
		# print(self.prior.shape)
		# print((np.matmul(X, self.likelihood)).shape)
		log_p = np.log(self.prior) + np.sum(np.log(np.matmul(X, self.likelihood)), 0) + \
				np.sum(np.log(np.matmul(X, self.likelihood)), 0)
		# log_p = np.log(self.prior) + np.sum(np.log(self.likelihood * X[:,None]), 0) + \
		# 		np.sum(np.log( (1 - self.likelihood) * (1 - X[:,None])), 0)
		log_p -= np.max(log_p)
		posterior = np.exp(log_p)
		posterior /= np.sum(posterior)
		return posterior	




# train_sample = 20
# test_sample = 7
# num_of_features = 8
# num_of_class = 2

# train_X, train_y = np.ones((train_sample, num_of_features)), convertToOneHot(np.zeros(train_sample), num_of_class)
# test_X, test_y = np.ones((test_sample, num_of_features)), convertToOneHot(np.zeros(test_sample), num_of_class)
# train_X, train_y = np.ones((train_sample, num_of_features)), np.zeros(train_sample)
# test_X, test_y = np.ones((test_sample, num_of_features)), np.zeros(test_sample)
# NB = CategoricalNB(num_of_features, num_of_class)
# NB.fit(train_X, train_y)
# predict_y = NB.predict(test_X)
# print(predict_y)





