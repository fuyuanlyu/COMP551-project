from models import logistic, naivebayes
from myutility import convertToOneHot
import numpy as np
from sklearn.linear_model import LogisticRegression

k_fold = 5

def main():
## add four dataset
	data_car = np.load('Datasets/car_data_cleaned.npy')
	data_iris = np.load('Datasets/iris_data_cleaned.npy')
	data_adult = np.load('Datasets/adult_data_cleaned.npy')
	data_inosphere = np.load('Datasets/ionosphere_cleaned.npy')
	
	print("car data shape:", data_car.shape)
	print("iris data shape:", data_iris.shape)
	print("adult data shape:", data_adult.shape)
	print("ionosphere data shape:", data_inosphere.shape)




	# data_x = np.load('Datasets/iris_data_features.npy', allow_pickle=True)
	# data_y = np.load('Datasets/iris_data_target.npy', allow_pickle=True)

	np.random.shuffle(data_car)
	num_samples = data_car.shape[0]
	data_car_x = data_car[:num_samples,:-1]
	data_car_y = data_car[:num_samples,-1]

	np.random.shuffle(data_iris)
	num_samples = data_iris.shape[0]
	data_iris_x = data_iris[:num_samples,:-1]
	data_iris_y = data_iris[:num_samples,-1]

	np.random.shuffle(data_adult)
	num_samples = data_adult.shape[0]
	data_adult_x = data_adult[:num_samples,:-1]
	data_adult_y = data_adult[:num_samples,-1]

	np.random.shuffle(data_inosphere)
	num_samples = data_inosphere.shape[0]
	data_inosphere_x = data_inosphere[:num_samples,:-1]
	data_inosphere_y = data_inosphere[:num_samples,-1]





	## calculate car data 

	step = data_car_x.shape[0] // k_fold
	# print(data_y)
	num_of_class = int(np.max(data_car_y)+1)
	
	acc_model_car = 0.
	for i in range(k_fold):
		# test_x, test_y = data_x[i:i+step], data_y[i:i+step]
		# train_x, train_y = np.concatenate((data_x[0:i], data_x[i+step:]), axis=0), np.concatenate((data_y[0:i], data_y[i+step:]), axis=0)
		test_x, test_y = data_car_x[i:i+step], data_car_y[i:i+step]
		train_x, train_y = np.concatenate((data_car_x[0:i], data_car_x[i+step:]), axis=0), convertToOneHot(np.concatenate((data_car_y[0:i], data_car_y[i+step:]), axis=0), num_of_class)
		
		# print((np.max(data_y)+1).astype(int))

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
		acc_model_iris += np.sum(predict_y == test_y) / test_y.shape[0]


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

		LR_adult.fit(train_x, train_y)

		predict_y = LR_adult.predict(test_x)
		# print(predict_y)
		# print(np.squeeze(test_y).shape)
		# print(np.sum(predict_y == np.squeeze(test_y)))
		acc_model_adult += np.sum(predict_y == test_y) / test_y.shape[0]


	acc_skl_adult = 0.
	for i in range(k_fold):
		test_x, test_y = data_adult_x[i:i+step], data_adult_y[i:i+step]
		train_x, train_y = np.concatenate((data_adult_x[0:i], data_adult_x[i+step:]), axis=0), np.concatenate((data_adult_y[0:i], data_adult_y[i+step:]), axis=0)
		
		LR_adult2 = LogisticRegression(solver='lbfgs', multi_class='multinomial')
		LR_adult2.fit(train_x, train_y)
		predict_y = LR_adult2.predict(test_x)
		# print(predict_y == test_y)
		acc_skl_adult += np.sum(predict_y == test_y) / test_y.shape[0]

## calculate inosphere data 
	
	step = data_inosphere_x.shape[0] // k_fold
	# print(data_y)
	num_of_class = int(np.max(data_inosphere_y)+1)
	
	acc_model_inosphere = 0.
	for i in range(k_fold):
		# test_x, test_y = data_x[i:i+step], data_y[i:i+step]
		# train_x, train_y = np.concatenate((data_x[0:i], data_x[i+step:]), axis=0), np.concatenate((data_y[0:i], data_y[i+step:]), axis=0)
		test_x, test_y = data_inosphere_x[i:i+step], data_inosphere_y[i:i+step]
		train_x, train_y = np.concatenate((data_inosphere_x[0:i], data_inosphere_x[i+step:]), axis=0), convertToOneHot(np.concatenate((data_inosphere_y[0:i], data_inosphere_y[i+step:]), axis=0), num_of_class)
		
		# print((np.max(data_y)+1).astype(int))

		LR_inosphere = logistic.logistic_regression(train_x.shape[1], train_y.shape[1])

		LR_inosphere.fit(train_x, train_y)

		predict_y = LR_inosphere.predict(test_x)
		# print(predict_y)
		# print(np.squeeze(test_y).shape)
		# print(np.sum(predict_y == np.squeeze(test_y)))
		acc_model_inosphere += np.sum(predict_y == test_y) / test_y.shape[0]


	acc_skl_inosphere = 0.
	for i in range(k_fold):
		test_x, test_y = data_inosphere_x[i:i+step], data_inosphere_y[i:i+step]
		train_x, train_y = np.concatenate((data_inosphere_x[0:i], data_inosphere_x[i+step:]), axis=0), np.concatenate((data_inosphere_y[0:i], data_inosphere_y[i+step:]), axis=0)
		
		LR_inosphere2 = LogisticRegression(solver='lbfgs', multi_class='multinomial')
		LR_inosphere2.fit(train_x, train_y)
		predict_y = LR_inosphere2.predict(test_x)
		# print(predict_y == test_y)
		acc_skl_inosphere += np.sum(predict_y == test_y) / test_y.shape[0]



	print("car data")
	print("skilearn model predicted accuracy:", acc_skl_car/k_fold)
	print("our model predicted accuracy:\n", acc_model_car/k_fold)
	print("")

	print("iris data")
	print("skilearn model predicted accuracy:", acc_skl_iris/k_fold)
	print("our model predicted accuracy:\n", acc_model_iris/k_fold)
	print("")
	
	print("adult data")
	print("skilearn model predicted accuracy:", acc_skl_adult/k_fold)
	print("our model predicted accuracy:\n", acc_model_adult/k_fold)
	print("")

	print("inosphere data")
	print("skilearn model predicted accuracy:", acc_skl_inosphere/k_fold)
	print("our model predicted accuracy:\n", acc_model_inosphere/k_fold)
	print("")




if __name__ == '__main__':
	main()





