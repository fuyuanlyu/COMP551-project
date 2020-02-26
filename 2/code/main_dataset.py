from sklearn.naive_bayes import MultinomialNB
from dataset.dataset import get_twenty_dataset, get_IMDB_dataset
import numpy as np

datasets = ['20 news group', 'IMDB Reviews']

def main():
	for dataset in datasets:
		if dataset == '20 news group':
			x_train, y_train, x_test, y_test = get_twenty_dataset()
		elif dataset == 'IMDB Reviews':
			x_train, y_train, x_test, y_test = get_IMDB_dataset()
		clf = MultinomialNB().fit(x_train, y_train)
		y_predicted = clf.predict(x_test)
		print(dataset + ':', np.mean(y_predicted == y_test))

if __name__ == '__main__':
	main()