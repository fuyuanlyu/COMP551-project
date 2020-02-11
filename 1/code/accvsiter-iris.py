from models import logistic, naivebayes
from models import myutility
import numpy as np
from matplotlib import pyplot as plt
# datasets = ['iris_data_cleaned', 'car_data_cleaned',\
#  'adult_data_cleaned', 'ionosphere_cleaned']
datasets = ['iris_data_cleaned']
for dataset in datasets:
	print("====",dataset,"====")
	data = np.load('Datasets/'+dataset+'.npy')
	np.random.shuffle(data)
	num_samples = data.shape[0]
	data_x = data[:num_samples,:-1]
	data_y = data[:num_samples,-1]
	iteration=2000
	k_fold = 5
	step = data_x.shape[0] // k_fold
	num_of_class = int(np.max(data_y)+1)
	train_acc_all= [0]*iteration
	test_acc_all=[0]*iteration
	train_accs_avg=[]*len(datasets)
	test_accs_avg=[]*len(datasets)
	for i in range(k_fold):
		train_accs = []
		test_accs = []
		test_x, test_y = data_x[i:i+step], data_y[i:i+step]
		train_x, train_y = np.concatenate((data_x[0:i], \
				data_x[i+step:]), axis=0), \
			myutility.convertToOneHot(np.concatenate((data_y[0:i], \
				data_y[i+step:]), axis=0), num_of_class)
		LR = logistic.logistic_regression(train_x.shape[1], train_y.shape[1], max_iter=1000,thres=0.05,lr=0.001)
		for i in range(iteration):
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
			# print(train_accs)
		train_acc_all = [train_acc_all[i] + train_accs[i] for i in range(len(train_accs))]
		test_acc_all = [test_acc_all[i] + test_accs[i] for i in range(len(test_accs))]

	train_acc_avg= [e/k_fold for e in train_acc_all]
	train_accs_avg.append(train_acc_avg)
	test_acc_avg = [e/k_fold for e in test_acc_all ]
	test_accs_avg.append(test_acc_avg)

# fig, axes= plt.subplots(2,2,figsize=(10,10))
# fig.subplots_adjust(bottom=-0.8)
plt.plot(train_accs_avg[0],label="train set")
plt.plot(test_accs_avg[0],label='validation set')
plt.legend(['train set','validation set'])
plt.xlabel('number of iterations')
plt.ylabel('Accuracy')
plt.title('iris')


plt.show()