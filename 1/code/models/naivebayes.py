import numpy as np
from models import myutility

epsilon = 1e-5


class NaiveBayes():
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

	def fit(self, X, y):
		# Initialize the likehood matrix
		# The matrix itself is dependent of the number possible categories of each features
		self.likelihood = []
		for i in range(self.num_of_features):
			max_dim_this_feature = (np.max(X[:,i])+1).astype(int)
			temp_likelihood = np.zeros((max_dim_this_feature, self.num_of_class))
			self.likelihood.append(temp_likelihood)

		# MLE the prior
		# The optimal for the prior is the number of samples in each class 
		# normalized by the number of total samples
		self.prior = np.sum(y, axis=0) / np.sum(y)

		# MLE the likelihood
		# The optimal for the likelihood is the number of samples in each 
		for i in range(self.num_of_features):
			temp_likelihood = self.likelihood[i]
			max_dim_this_feature = int(np.max(X[:,i])+1)
			for j in range(X.shape[0]):
				temp_likelihood[int(X[j,i]), np.squeeze(np.argwhere(y[j]==1))] += 1
			normal = np.sum(temp_likelihood, axis=1) + epsilon
			temp_likelihood /= normal[:, None]
			self.likelihood[i] = temp_likelihood
		return self

	def predict(self, X):
		posterior = np.zeros((X.shape[0], self.num_of_class))
		for sample in range(X.shape[0]):
			for i in range(self.num_of_class):
				posterior[sample, i] = self.prior[i] 
				for j in range(self.num_of_features):
					temp_likelihood = self.likelihood[j]
					print(X[i,j], i)
					posterior[sample, i] *= temp_likelihood[int(X[i,j]), i]

		normal = np.sum(posterior, axis=1) + epsilon
		posterior /= normal[:,None]
		y = np.argmax(posterior, axis=1)

		return y



## Testing

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





