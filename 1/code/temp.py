from models import logistic, naivebayes
from models import myutility
import numpy as np

data = np.load('Datasets/ionosphere_cleaned.npy')
np.random.shuffle(data)
num_samples = data.shape[0]
data_x = data[:num_samples,:-1]
data_y = data[:num_samples,-1]

k_fold = 5
step = data_x.shape[0] // k_fold
num_of_class = int(np.max(data_y)+1)

for i in range(k_fold):
	train_accs = []
	test_accs = []
	test_x, test_y = data_x[i:i+step], data_y[i:i+step]
	train_x, train_y = np.concatenate((data_x[0:i], \
			data_x[i+step:]), axis=0), \
		myutility.convertToOneHot(np.concatenate((data_y[0:i], \
			data_y[i+step:]), axis=0), num_of_class)
	LR = logistic.logistic_regression(train_x.shape[1], train_y.shape[1], max_iter=100)
	for i in range(100):
		LR.fit(train_x, train_y)
		train_predict_y = LR.predict(train_x)
		# if i == 0:
		# 	print(train_predict_y.shape)
		# 	print(train_y.shape)
		# 	print(train_predict_y == train_y)
		test_predict_y = LR.predict(test_x)
		train_acc = np.sum(np.argmax(train_y, axis=1) == train_predict_y) / train_y.shape[0]
		test_acc = np.sum(test_y == test_predict_y) / test_y.shape[0]
		train_accs.append(train_acc)
		test_accs.append(test_acc)

	print(train_accs)
