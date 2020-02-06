from models import logistic, naivebayes
from myutility import convertToOneHot
import numpy as np

k_fold = 5

def main():
	# data = np.load('Datasets/adult_data_cleaned.npy')
	data = np.load('Datasets/iris_data_cleaned.npy')
	np.random.shuffle(data)
	num_samples = data.shape[0]
	data_x = data[:num_samples,:-1]
	data_y = data[:num_samples,-1:]

	step = data_x.shape[0] // k_fold
	num_of_class = int(np.max(data_y)+1)
	acc = 0.

	for i in range(k_fold):
		# test_x, test_y = data_x[i:i+step], data_y[i:i+step]
		# train_x, train_y = np.concatenate((data_x[0:i], data_x[i+step:]), axis=0), np.concatenate((data_y[0:i], data_y[i+step:]), axis=0)
		test_x, test_y = data_x[i:i+step], data_y[i:i+step]
		train_x, train_y = np.concatenate((data_x[0:i], data_x[i+step:]), axis=0), convertToOneHot(np.concatenate((data_y[0:i], data_y[i+step:]), axis=0), num_of_class)
		
		# print((np.max(data_y)+1).astype(int))

		LR = logistic.logistic_regression(train_x.shape[1], train_y.shape[1])

		LR.fit(train_x, train_y)

		predict_y = LR.predict(test_x)
		# print(predict_y.shape)
		# print(np.squeeze(test_y).shape)
		# print(np.sum(predict_y == np.squeeze(test_y)))
		acc += np.sum(predict_y == np.squeeze(test_y)) / test_y.shape[0]
		print(np.sum(predict_y == np.squeeze(test_y)) / test_y.shape[0])
	print(acc/k_fold)


if __name__ == '__main__':
	main()




