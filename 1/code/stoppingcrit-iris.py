from models import logistic, naivebayes
from models import myutility
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from matplotlib import pyplot as plt

k_fold = 5

# datasets = ['iris_data_cleaned', 'car_data_cleaned',\
#  'adult_data_cleaned', 'ionosphere_cleaned']
datasets = ['iris_data_cleaned']

def main():
    # fig, axes= plt.subplots(2,2,figsize=(10,10))
    # fig.subplots_adjust(bottom=-0.8)
    acc_all=[]*len(datasets)
    size_of_datasets=[]*len(datasets)
    ths=0.001*np.arange(10,1000,10)
    for dataset in datasets:
    	print("===", dataset, "===")
    	print("Predicting based on logistic regression")
    	accs=main_lr(dataset,range_th=ths)
    	acc_all.append(accs)

    plt.plot(ths,acc_all[0])
    plt.xlabel('stopping threshold')
    plt.ylabel('Accuracy')
    plt.title('iris')

    # axes[0,1].plot(ths,acc_all[1],label="LR")
    # axes[0,1].set_xlabel('stopping threshold')
    # axes[0,1].set_ylabel('Accuracy')
    # axes[0,1].set_title('car')
	#
    # axes[1,0].plot(ths,acc_all[2])
    # axes[1,0].set_xlabel('stopping threshold')
    # axes[1,0].set_ylabel('Accuracy')
    # axes[1,0].set_title('adult')
	#
	#
    # axes[1,1].plot(ths,acc_all[2],label="LR")
    # axes[1,1].set_xlabel('stopping threshold')
    # axes[1,1].set_ylabel('Accuracy')
    # axes[1,1].set_title('inosphere')

    plt.show()





## Predicting based on logistic regression
def main_lr(dataset,range_th):
	data = np.load('Datasets/' + dataset + '.npy')
	np.random.shuffle(data)
	num_samples = data.shape[0]
	data_x = data[:num_samples,:-1]
	data_y = data[:num_samples,-1]


	accs=[]
	ths=[]
	for th in range_th:
	    step = data_x.shape[0] // k_fold
	    num_of_class = int(np.max(data_y)+1)
	    acc = 0.       
	    for i in range(k_fold):
	        test_x, test_y = data_x[i:i+step], data_y[i:i+step]
	        train_x, train_y = np.concatenate((data_x[0:i], \
	                data_x[i+step:]), axis=0), \
	            myutility.convertToOneHot(np.concatenate((data_y[0:i], \
	                data_y[i+step:]), axis=0), num_of_class)

	        LR = logistic.logistic_regression(train_x.shape[1], train_y.shape[1],thres=th/1000)
	        LR.fit(train_x, train_y)
	        predict_y = LR.predict(test_x)
	        acc += np.sum(predict_y == test_y) / test_y.shape[0]
	    accs.append(acc/k_fold)
	    ths.append(th/1000)
	
	return(accs)
	#     # print(acc/k_fold)

	# plt.plot(ths,accs)
	# plt.xlabel("stopping threshold")
	# plt.ylabel("accuracy")
	# plt.show()




if __name__ == '__main__':
	main()





