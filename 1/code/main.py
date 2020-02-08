from models import logistic, naivebayes
from models import myutility
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

k_fold = 5

def main():
	# data = np.load('Datasets/adult_data_cleaned.npy')
	data = np.load('Datasets/iris_data_cleaned.npy')
	# data = np.load('Datasets/car_data_cleaned.npy')
	# data = np.load('Datasets/ionosphere_data_cleaned.npy')
	# data_x = np.load('Datasets/iris_data_features.npy', allow_pickle=True)
	# data_y = np.load('Datasets/iris_data_target.npy', allow_pickle=True)

	step = data_x.shape[0] // k_fold
	# print(data_x)
	num_of_class = int(np.max(data_y)+1)
	acc = 0.

		LR_car = logistic.logistic_regression(train_x.shape[1], train_y.shape[1])

		LR_car.fit(train_x, train_y)

		predict_y = LR_car.predict(test_x)
		# print(predict_y)
		# print(np.squeeze(test_y).shape)
		# print(np.sum(predict_y == np.squeeze(test_y)))
		acc_model_car += np.sum(predict_y == test_y) / test_y.shape[0]


	acc_skl_car = 0.
	for i in range(k_fold):
		test_x, test_y = data_car_x[i:i+step], data_car_y[i:i+step]
		train_x, train_y = np.concatenate((data_car_x[0:i], data_car_x[i+step:]), axis=0), np.concatenate((data_car_y[0:i], data_car_y[i+step:]), axis=0)
		
		LR_car2 = LogisticRegression(solver='lbfgs', multi_class='multinomial')
		LR_car2.fit(train_x, train_y)
		predict_y = LR_car2.predict(test_x)
		# print(predict_y == test_y)
		acc_skl_car += np.sum(predict_y == test_y) / test_y.shape[0]


## calculate iris data 
	
	step = data_iris_x.shape[0] // k_fold
	# print(data_y)
	num_of_class = int(np.max(data_iris_y)+1)
	
	acc_model_iris = 0.
	for i in range(k_fold):
		# test_x, test_y = data_x[i:i+step], data_y[i:i+step]
		# train_x, train_y = np.concatenate((data_x[0:i], data_x[i+step:]), axis=0), np.concatenate((data_y[0:i], data_y[i+step:]), axis=0)
		test_x, test_y = data_iris_x[i:i+step], data_iris_y[i:i+step]
		train_x, train_y = np.concatenate((data_iris_x[0:i], data_iris_x[i+step:]), axis=0), convertToOneHot(np.concatenate((data_iris_y[0:i], data_iris_y[i+step:]), axis=0), num_of_class)
		
		# print((np.max(data_y)+1).astype(int))

		LR_iris = logistic.logistic_regression(train_x.shape[1], train_y.shape[1])

		LR_iris.fit(train_x, train_y)

		predict_y = LR_iris.predict(test_x)
		# print(predict_y)
		# print(np.squeeze(test_y).shape)
		# print(np.sum(predict_y == np.squeeze(test_y)))
		acc += np.sum(predict_y == test_y) / test_y.shape[0]
	print(acc/k_fold)
	# print(predict_y)
	# print(test_y)


	acc_skl_iris = 0.
	for i in range(k_fold):
		test_x, test_y = data_iris_x[i:i+step], data_iris_y[i:i+step]
		train_x, train_y = np.concatenate((data_iris_x[0:i], data_iris_x[i+step:]), axis=0), np.concatenate((data_iris_y[0:i], data_iris_y[i+step:]), axis=0)
		
		LR_iris2 = LogisticRegression(solver='lbfgs', multi_class='multinomial')
		LR_iris2.fit(train_x, train_y)
		predict_y = LR_iris2.predict(test_x)
		# print(predict_y == test_y)
		acc_skl_iris += np.sum(predict_y == test_y) / test_y.shape[0]

## calculate adult data 
	
	step = data_adult_x.shape[0] // k_fold
	# print(data_y)
	num_of_class = int(np.max(data_adult_y)+1)
	
	acc_model_adult = 0.
	for i in range(k_fold):
		# test_x, test_y = data_x[i:i+step], data_y[i:i+step]
		# train_x, train_y = np.concatenate((data_x[0:i], data_x[i+step:]), axis=0), np.concatenate((data_y[0:i], data_y[i+step:]), axis=0)
		test_x, test_y = data_adult_x[i:i+step], data_adult_y[i:i+step]
		train_x, train_y = np.concatenate((data_adult_x[0:i], data_adult_x[i+step:]), axis=0), convertToOneHot(np.concatenate((data_adult_y[0:i], data_adult_y[i+step:]), axis=0), num_of_class)
		
		# print((np.max(data_y)+1).astype(int))

		LR_adult = logistic.logistic_regression(train_x.shape[1], train_y.shape[1])

def main2():
	# data = np.load('Datasets/adult_data_cleaned.npy')
	data = np.load('Datasets/iris_data_cleaned.npy')
	# data = np.load('Datasets/car_data_cleaned.npy')
	# data = np.load('Datasets/ionosphere_data_cleaned.npy')
	# data_x = np.load('Datasets/iris_data_features.npy', allow_pickle=True)
	# data_y = np.load('Datasets/iris_data_target.npy', allow_pickle=True)
	np.random.shuffle(data)
	num_samples = data.shape[0]
	data_x = data[:num_samples,:-1]
	data_y = data[:num_samples,-1]

	step = data_x.shape[0] // k_fold
	# print(data_x)
	num_of_class = int(np.max(data_y)+1)
	print(num_of_class)
	acc = 0.

	for i in range(k_fold):
		# test_x, test_y = data_x[i:i+step], data_y[i:i+step]
		# train_x, train_y = np.concatenate((data_x[0:i], data_x[i+step:]), axis=0), np.concatenate((data_y[0:i], data_y[i+step:]), axis=0)
		test_x, test_y = data_x[i:i+step], data_y[i:i+step]
		train_x, train_y = np.concatenate((data_x[0:i], data_x[i+step:]), axis=0), myutility.convertToOneHot(np.concatenate((data_y[0:i], data_y[i+step:]), axis=0), num_of_class)
		
		# print((np.max(data_y)+1).astype(int))

		NB = naivebayes.NaiveBayes(train_x.shape[1], train_y.shape[1])

		NB.fit(train_x, train_y)

		predict_y = NB.predict(test_x)
		# print(predict_y)
		# print(np.squeeze(test_y).shape)
		# print(np.sum(predict_y == np.squeeze(test_y)))
		print(predict_y)
		print(test_y)
		acc += np.sum(predict_y == test_y) / test_y.shape[0]
	print(acc/k_fold)
	# print(predict_y)
	# print(test_y)

	acc = 0.
	for i in range(k_fold):
		test_x, test_y = data_x[i:i+step], data_y[i:i+step]
		train_x, train_y = np.concatenate((data_x[0:i], data_x[i+step:]), axis=0), np.concatenate((data_y[0:i], data_y[i+step:]), axis=0)
		
		NB2 = MultinomialNB()
		NB2.fit(train_x, train_y)
		predict_y = NB2.predict(test_x)
		# print(predict_y == test_y)
		acc += np.sum(predict_y == test_y) / test_y.shape[0]
	print(acc/k_fold)




if __name__ == '__main__':
	# main()
	main2()





