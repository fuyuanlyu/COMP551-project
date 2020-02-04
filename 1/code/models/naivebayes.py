import numpy as np

class BernoulliNB():
	def fit(self, X, y):
		a = X

	def predict(self, X):
		y = 0
		return y


class MultinomialNB():
	def fit(self, X, y):
		a = X

	def predict(self, X):
		y = 0
		return y	


train_sample = 20
test_sample = 7
num_of_features = 8

train_X, train_y = np.ones((train_sample, num_of_features)), np.zeros(train_sample)
test_X, test_y = np.ones((test_sample, num_of_features)), np.zeros(test_sample)
NB = BernoulliNB()
NB.fit(train_X, train_y)
predict_y = NB.predict(test_X)
print(predict_y)





