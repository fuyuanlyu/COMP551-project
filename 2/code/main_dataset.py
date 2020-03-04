from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import  AdaBoostClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from dataset.dataset import get_twenty_dataset, get_IMDB_dataset

import numpy as np
##overall performance is IMDB>news group
# datasets = ['20 news group', 'IMDB Reviews']


def main(x_train, y_train, x_test, y_test):
	## multinomialNB model
	clf_NB = MultinomialNB().fit(x_train, y_train)
	print('multinomialNB model')
	y_predicted_NB = clf_NB.predict(x_test)
	#print(dataset + ':', np.mean(y_predicted_NB == y_test))
	print(np.mean(y_predicted_NB == y_test))

	## logistic regression model
	clf_LR = LogisticRegression(random_state=0).fit(x_train, y_train)
	print('logistic regression model')
	y_predicted_LR = clf_LR.predict(x_test)
	#print(dataset + ':', np.mean(y_predicted_LR == y_test))
	print(np.mean(y_predicted_LR == y_test))

	clf_DT = DecisionTreeClassifier(random_state=0,max_depth=100,criterion="gini",min_samples_leaf=2).fit(x_train, y_train) ##can change the depth to increase acc, gini is better,mini_sample_leaf increase, acc get down
	print('Decision Tree model')
	y_predicted_DT = clf_DT.predict(x_test)
	#print(dataset + ':', np.mean(y_predicted_DT == y_test))
	print(np.mean(y_predicted_DT == y_test))

	clf_SVC = LinearSVC(random_state=0,tol=1e-05).fit(x_train, y_train)
	print('SVM model')
	y_predicted_SVC = clf_SVC.predict(x_test)
	#print(dataset + ':', np.mean(y_predicted_SVC == y_test))
	print(np.mean(y_predicted_SVC == y_test))

	clf_ADB = AdaBoostClassifier(n_estimators=100,random_state=0,learning_rate=1).fit(x_train, y_train)   ## there is trade off between n_estimator and lr, but i cannot figure our how to increase acc
	print('AdaBoost model')
	y_predicted_ADB = clf_ADB.predict(x_test)
	#print(dataset + ':', np.mean(y_predicted_ADB == y_test))
	print(np.mean(y_predicted_ADB == y_test))

	clf_RDF = RandomForestClassifier(max_depth=40,random_state=0,n_estimators=100).fit(x_train, y_train) ## you can change the max_depth to increase acc, but when it's bigger than approximate 40 it will not change much
	print('Random forest model')
	y_predicted_RDF = clf_RDF.predict(x_test)
	#print(dataset + ':', np.mean(y_predicted_RDF == y_test))
	print(np.mean(y_predicted_RDF == y_test))

	clf_NN = MLPClassifier(solver='lbfgs',hidden_layer_sizes=(30,1024),max_iter=5000)  ##trade off between hidden units and layer depth
	print('MLPClassifier model')
	clf_NN.fit(x_train, y_train)
	y_predicted_NN = clf_NN.predict(x_test)
	#print(dataset + ':', np.mean(y_predicted_RDF == y_test))
	print(np.mean(y_predicted_NN == y_test))

	clf_xgb = XGBClassifier(learning_rate=0.01)
	print('XGboost model')
	clf_xgb.fit(x_train, y_train)
	y_predicted_XGB = clf_xgb.predict(x_test)
	#print(dataset + ':', np.mean(y_predicted_RDF == y_test))
	print(np.mean(y_predicted_XGB == y_test))


	return {'NB':clf_NB,'LR':clf_LR,'DT':clf_DT,'SVC':clf_SVC,'ADB':clf_ADB,'RDF':clf_RDF,'NN':clf_NN,'XGB':clf_xgb}


if __name__ == '__main__':
	datasets = ['20 news group','IMDB Reviews']
	for dataset in datasets:
		if dataset == '20 news group':
			x_train, y_train, x_test, y_test = get_twenty_dataset()
		elif dataset == 'IMDB Reviews':
			x_train, y_train, x_test, y_test = get_IMDB_dataset()
		main(x_train, y_train, x_test, y_test)