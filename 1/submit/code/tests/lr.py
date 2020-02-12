import sys
sys.path.append("..")
from models import logistic, naivebayes
from models import myutility
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from matplotlib import pyplot as plt

k_fold = 5

datasets = ['iris_data_cleaned', 'car_data_cleaned',\
 'adult_data_cleaned', 'ionosphere_cleaned']


def main():
    fig,axes=plt.subplots(2,2,figsize=(5,2))
    fig.subplots_adjust(bottom=-0.6)
    acc_all=[]*len(datasets)
    for dataset in datasets:
        print("===", dataset, "===")
        print("Predicting based on logistic regression")
        accs=[]
        lrs=[]
        for lr in range(1,50,1): ##here we can change the learning rate range
            acc=main_lr(dataset,lr/1000)
            accs.append(acc)
            lrs.append(lr/1000)

        acc_all.append(accs)
    axes[0,0].plot(lrs,acc_all[0])
    axes[0,0].set_xlabel('learning rate')
    axes[0,0].set_ylabel('Accuracy')
    axes[0,0].set_title('iris')
    axes[0,1].plot(lrs,acc_all[1])
    axes[0,1].set_xlabel('learning rate')
    axes[0,1].set_ylabel('Accuracy')
    axes[0,1].set_title('car')
    axes[1,0].plot(lrs,acc_all[2])
    axes[1,0].set_xlabel('learning rate')
    axes[1,0].set_ylabel('Accuracy')
    axes[1,0].set_title('adult')
    axes[1,1].plot(lrs,acc_all[3])
    axes[1,1].set_xlabel('learning rate')
    axes[1,1].set_ylabel('Accuracy')
    axes[1,1].set_title('inosphere')
    plt.show()
        # plt.plot(lrs,accs) 
        # plt.show()   


## Predicting based on logistic regression
def main_lr(dataset,lr):
	data = np.load('../Datasets/' + dataset + '.npy')
	np.random.shuffle(data)
	num_samples = data.shape[0]
	data_x = data[:num_samples,:-1]
	data_y = data[:num_samples,-1]

	step = data_x.shape[0] // k_fold
	num_of_class = int(np.max(data_y)+1)
	acc = 0.

	for i in range(k_fold):
		test_x, test_y = data_x[i:i+step], data_y[i:i+step]
		train_x, train_y = np.concatenate((data_x[0:i], \
				data_x[i+step:]), axis=0), \
			myutility.convertToOneHot(np.concatenate((data_y[0:i], \
				data_y[i+step:]), axis=0), num_of_class)

		LR = logistic.logistic_regression(train_x.shape[1], train_y.shape[1],100,lr)
		LR.fit(train_x, train_y)
		predict_y = LR.predict(test_x)
		acc += np.sum(predict_y == test_y) / test_y.shape[0]
	
	return(acc/k_fold)

	# acc = 0.
	# for i in range(k_fold):
	# 	test_x, test_y = data_x[i:i+step], data_y[i:i+step]
	# 	train_x, train_y = np.concatenate((data_x[0:i], \
	# 			data_x[i+step:]), axis=0), \
	# 		np.concatenate((data_y[0:i], data_y[i+step:]), axis=0)
		
	# 	LR2 = LogisticRegression(solver='lbfgs', multi_class='auto',l1_ratios=0.001)
	# 	LR2.fit(train_x, train_y)
	# 	predict_y = LR2.predict(test_x)
	# 	acc += np.sum(predict_y == test_y) / test_y.shape[0]
	# print(acc/k_fold)




if __name__ == '__main__':
	main()





