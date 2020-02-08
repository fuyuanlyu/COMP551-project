from models import logistic, naivebayes
from models import myutility
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

k_fold = 5

def main_lr():
	data = np.load('Datasets/iris_data_cleaned.npy')
	np.random.shuffle(data)
	num_samples = data.shape[0]
	data_x = data[:num_samples,:-1]
	data_y = data[:num_samples,-1]

	step = data_x.shape[0] // k_fold
	num_of_class = int(np.max(data_y)+1)
	acc = 0.

	for i in range(k_fold):
		test_x, test_y = data_x[i:i+step], data_y[i:i+step]
		train_x, train_y = np.concatenate((data_x[0:i], data_x[i+step:]), axis=0), myutility.convertToOneHot(np.concatenate((data_y[0:i], data_y[i+step:]), axis=0), num_of_class)

		LR = logistic.logistic_regression(train_x.shape[1], train_y.shape[1])
		LR.fit(train_x, train_y)
		predict_y = LR.predict(test_x)
		acc += np.sum(predict_y == test_y) / test_y.shape[0]
	print(acc/k_fold)

	acc = 0.
	for i in range(k_fold):
		test_x, test_y = data_x[i:i+step], data_y[i:i+step]
		train_x, train_y = np.concatenate((data_x[0:i], data_x[i+step:]), axis=0), np.concatenate((data_y[0:i], data_y[i+step:]), axis=0)
		
		LR2 = LogisticRegression(solver='lbfgs', multi_class='auto')
		LR2.fit(train_x, train_y)
		predict_y = LR2.predict(test_x)
		acc += np.sum(predict_y == test_y) / test_y.shape[0]
	print(acc/k_fold)


def main_nb():
	data = np.load('Datasets/iris_data_cleaned.npy')
	np.random.shuffle(data)
	num_samples = data.shape[0]
	data_x = data[:num_samples,:-1]
	data_y = data[:num_samples,-1]

	step = data_x.shape[0] // k_fold
	num_of_class = int(np.max(data_y)+1)
	acc = 0.

	for i in range(k_fold):
		test_x, test_y = data_x[i:i+step], data_y[i:i+step]
		train_x, train_y = np.concatenate((data_x[0:i], data_x[i+step:]), axis=0), myutility.convertToOneHot(np.concatenate((data_y[0:i], data_y[i+step:]), axis=0), num_of_class)
		
		NB = naivebayes.NaiveBayes(train_x.shape[1], train_y.shape[1])
		NB.fit(train_x, train_y)
		predict_y = NB.predict(test_x)
		acc += np.sum(predict_y == test_y) / test_y.shape[0]
	print(acc/k_fold)

	acc = 0.
	for i in range(k_fold):
		test_x, test_y = data_x[i:i+step], data_y[i:i+step]
		train_x, train_y = np.concatenate((data_x[0:i], data_x[i+step:]), axis=0), np.concatenate((data_y[0:i], data_y[i+step:]), axis=0)
		
		NB2 = MultinomialNB()
		NB2.fit(train_x, train_y)
		predict_y = NB2.predict(test_x)
		acc += np.sum(predict_y == test_y) / test_y.shape[0]
	print(acc/k_fold)




if __name__ == '__main__':
	main_lr()
	main_nb()





