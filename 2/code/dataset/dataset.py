import numpy as np
import tarfile
import pickle
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.decomposition import TruncatedSVD
import string
from nltk.corpus import stopwords

# Get 20 news group dataset
def get_twenty_dataset(remove_stop_word=False, use_PCA=False, n_components=2):
	twenty_train = fetch_20newsgroups(subset='train', \
		remove=['headers', 'footers', 'quote'], shuffle=True)
	twenty_test = fetch_20newsgroups(subset='test', \
		remove=['headers', 'footers', 'quote'], shuffle=True)

	if remove_stop_word:
		count_vect = CountVectorizer(stop_words=stopwords.words('english') + list(string.punctuation))
	else:
		count_vect = CountVectorizer()
	X_train_counts = count_vect.fit_transform(twenty_train.data)
	X_test_counts = count_vect.transform(twenty_test.data)

	tfidf_transformer = TfidfTransformer()
	X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
	X_test_tfidf = tfidf_transformer.transform(X_test_counts)

	if use_PCA:
		pca = TruncatedSVD(n_components = n_components) 
		X_train_tfidf = pca.fit_transform(X_train_tfidf)
		X_test_tfidf = pca.transform(X_test_tfidf)

	return X_train_tfidf, twenty_train.target, X_test_tfidf, twenty_test.target


# Get IMDB Review dataset
def get_IMDB_dataset(remove_stop_word=False, use_PCA=False, n_components=2):
	with open('./dataset/IMDB.pickle', 'rb') as data:
		dataset = pickle.load(data)
	train_x_raw, train_y = dataset['train']
	test_x_raw, test_y = dataset['test']
	
	if remove_stop_word:
		count_vect = CountVectorizer(stop_words=stopwords.words('english') + list(string.punctuation))
	else:
		count_vect = CountVectorizer()
	X_train_counts = count_vect.fit_transform(train_x_raw)
	X_test_counts = count_vect.transform(test_x_raw)

	tfidf_transformer = TfidfTransformer()
	X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
	X_test_tfidf = tfidf_transformer.transform(X_test_counts)

	if use_PCA:
		pca = TruncatedSVD(n_components = n_components) 
		X_train_tfidf = pca.fit_transform(X_train_tfidf)
		X_test_tfidf = pca.transform(X_test_tfidf)

	return X_train_tfidf, train_y, X_test_tfidf, test_y

# Extract the txt files and assemble into a pickle dataset
def prepare_IMDB_dataset():
	PATH = "C:/Users/Thinkpad/Downloads/"
	tar = tarfile.open("./dataset/aclImdb_v1.tar.gz", "r")
	names = tar.getnames()
	train_x, train_y = [], []
	test_x, test_y = [], []

	for name in names:
		if 'train/pos/' in name:
			text = read_txt(PATH + name)
			train_x.append(text)
			train_y.append(1)
		elif 'train/neg/' in name:
			text = read_txt(PATH + name)
			train_x.append(text)
			train_y.append(0)
		elif 'test/pos/' in name:
			text = read_txt(PATH + name)
			test_x.append(text)
			test_y.append(1)
		elif 'test/neg/' in name:
			text = read_txt(PATH + name)
			test_x.append(text)
			test_y.append(0)
	tar.close()

	dataset = {"train": (train_x, train_y), 'test': (test_x, test_y)}
	with open('./dataset/IMDB.pickle', 'wb') as output:
		pickle.dump(dataset, output)


def read_txt(file_path):
	with open(file_path, encoding='utf-8') as file:
		try:
			text = file.read()
		except:
			print(file_path)
	return text



## Run it only once to extract txt files ##
# prepare_IMDB_dataset()





