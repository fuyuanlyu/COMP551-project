## intend to plot ROC curve, but it can only apply to binary classification, don't know how to calculate the
import sys
sys.path.append("..")
from models import logistic, naivebayes
from models import myutility
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
k_fold = 5

datasets = ['iris_data_cleaned', 'car_data_cleaned',\
 'adult_data_cleaned', 'ionosphere_cleaned']
# datasets = ['iris_data_cleaned']

def main():
	for dataset in datasets:
		print("===", dataset, "===")
		print("Predicting based on logistic regression")
		y_score,y_test,n_classes= main_lr(dataset)
		# print(y_score)
		# print(y_test)
		# fpr=dict()
		# tpr=dict()
		# roc_auc=dict()
		# for i in range(n_classes-1):
		# 	fpr[i],tpr[i],_ = roc_curve(y_test[:,i],y_score[:,i])
		# 	roc_auc[i]=auc(fpr[i],tpr[i])
##plot roc
        # all_fpr=np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
        # mean_tpr=np.zeros_like(all_fpr)
        # for i in range(n_classes):
        #     mean_tpr += interp(all_fpr,fpr[i],tpr[i])
        # mean_tpr /= n_classes
        # fpr["macro"]=all_fpr
        # tpr["macro"]=mean_tpr
        # # roc_auc["macro"]=auc(fpr["macro"],tpr["macro"])

		# fpr["micro"],tpr["micro"],_ = roc_curve(y_test.ravel(),y_score.ravel())
		# roc_auc["micro"]=auc(fpr["micro"],tpr["micro"])
        
		# plt.plot(fpr["micro"],tpr["micro"],label='micro-average ROC curve')
		# plt.show()

		# print('Predicting based on Naive Bayes')
		# main_nb(dataset)

## Predicting based on logistic regression
def main_lr(dataset):
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

		LR = logistic.logistic_regression(train_x.shape[1], train_y.shape[1])
		LR.fit(train_x, train_y)
		predict_y = LR.predict(test_x)
		acc += np.sum(predict_y == test_y) / test_y.shape[0]
	print(acc/k_fold)
	print(metrics.classification_report(test_y,predict_y))

	acc = 0.
	for i in range(k_fold):
		test_x, test_y = data_x[i:i+step], data_y[i:i+step]
		train_x, train_y = np.concatenate((data_x[0:i], \
				data_x[i+step:]), axis=0), \
			np.concatenate((data_y[0:i], data_y[i+step:]), axis=0)
		
		LR2 = LogisticRegression(solver='lbfgs', multi_class='auto')
		LR2.fit(train_x, train_y)
		predict_y = LR2.predict(test_x)
		acc += np.sum(predict_y == test_y) / test_y.shape[0]
	print(acc/k_fold)
	print(metrics.classification_report(test_y,predict_y))

	return predict_y,test_y,num_of_class

## Predicting based on naive bayes
def main_nb(dataset):
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

		NB = naivebayes.NaiveBayes(train_x.shape[1], train_y.shape[1])
		NB.fit(train_x, train_y)
		predict_y = NB.predict(test_x)
		acc += np.sum(predict_y == test_y) / test_y.shape[0]
	print(acc/k_fold)
	print(metrics.classification_report(test_y,predict_y))

	acc = 0.
	for i in range(k_fold):
		test_x, test_y = data_x[i:i+step], data_y[i:i+step]
		train_x, train_y = np.concatenate((data_x[0:i], data_x[i+step:]), axis=0), \
			np.concatenate((data_y[0:i], data_y[i+step:]), axis=0)
		
		NB2 = MultinomialNB()
		NB2.fit(train_x, train_y)
		predict_y = NB2.predict(test_x)
		acc += np.sum(predict_y == test_y) / test_y.shape[0]
	print(acc/k_fold)
	print(metrics.classification_report(test_y,predict_y))




if __name__ == '__main__':
	main()





