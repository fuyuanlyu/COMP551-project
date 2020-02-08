import numpy as np
# from myutility import convertToOneHot
from models import myutility

epsilon = 1e-5

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
		# print(max_dim_this_feature)
		# print(temp_likelihood)

		# MLE the prior
		# The optimal for the prior is the number of samples in each class 
		# normalized by the number of total samples
		self.prior = np.sum(y, axis=0) / np.sum(y)
		# print(self.prior)

		# MLE the likelihood
		# The optimal for the likelihood is the number of samples in each 
		
		# for i in range(self.num_of_features):
		# 	temp_likelihood = self.likelihood[i]
		# 	max_dim_this_feature = int(np.max(X[:,i])+1)
		# 	print(max_dim_this_feature)
		# 	print(convertToOneHot(X[:,i], max_dim_this_feature))
		# 	temp_likelihood = np.sum(convertToOneHot(X[:,i], max_dim_this_feature),axis=0)
		# 	print(temp_likelihood)
		# 	self.likelihood[i] = temp_likelihood

		for i in range(self.num_of_features):
			temp_likelihood = self.likelihood[i]
			max_dim_this_feature = int(np.max(X[:,i])+1)
			for j in range(X.shape[0]):
				# print(X[j,i], np.squeeze(np.argwhere(y[j]==1)), temp_likelihood.shape)
				temp_likelihood[X[j,i], np.squeeze(np.argwhere(y[j]==1))] += 1
			normal = np.sum(temp_likelihood, axis=1) + epsilon
			temp_likelihood /= normal[:, None]
			# temp_likelihood /= np.sum(temp_likelihood, axis=1) + epsilon
			self.likelihood[i] = temp_likelihood
		return self

	def predict(self, X):
		posterior = np.zeros((X.shape[0], self.num_of_class))
		for sample in range(X.shape[0]):
			for i in range(self.num_of_class):
				posterior[sample, i] = self.prior[i] 
				for j in range(self.num_of_features):
					# max_dim_this_feature = self.likelihood[j].shape[0]
					temp_likelihood = self.likelihood[j]
					# print(X[i,j])
					# print(temp_likelihood)
					posterior[sample, i] *= temp_likelihood[int(X[i,j]), i]
					# print(posterior)

		# print(posterior)
		normal = np.sum(posterior, axis=1) + epsilon
		posterior /= normal[:,None]
		# posterior /= np.sum(posterior, axis=1) + epsilon

		y = np.argmax(posterior, axis=1)

		return y


# train_sample = 20
# test_sample = 4
# num_of_features = 8
# num_of_class = 3

# train_X, train_y = np.random.randint(3, size=(train_sample, num_of_features)), \
# 	convertToOneHot(np.random.randint(num_of_class, size=train_sample), num_of_class)
# test_X, test_y = np.random.randint(3, size=(test_sample, num_of_features)),  \
# 	convertToOneHot(np.random.randint(num_of_class, size=test_sample), num_of_class)
# # train_X, train_y = np.ones((train_sample, num_of_features)), np.zeros(train_sample)
# # test_X, test_y = np.ones((test_sample, num_of_features)), np.zeros(test_sample)
# NB = NaiveBayes(num_of_features, num_of_class)
# NB.fit(train_X, train_y)
# predict_y = NB.predict(test_X)
# # print(predict_y)





