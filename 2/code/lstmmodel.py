from sklearn.datasets import fetch_20newsgroups
# Load tools we need for preprocessing
import urllib3
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from keras.utils import to_categorical
import os
from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation, Embedding


class LSTM_model():
    def __init__(self,vocab_size = 20000,max_length = 100,embedding_dim = 100):
        self.vocab_size=vocab_size
        self.max_length=max_length
        self.embedding_dim=embedding_dim

    # def main_train(self):
        ##process trainning dataset
        twenty_train = fetch_20newsgroups(subset='train', remove=['headers', 'footers', 'quote'], shuffle=True)
        texts = twenty_train.data # Extract text
        target = twenty_train.target # Extract target
        tokenizer_train = Tokenizer(num_words=self.vocab_size) # Setup tokenizer
        tokenizer_train.fit_on_texts(texts)
        sequences = tokenizer_train.texts_to_sequences(texts) # Generate sequences
        data = pad_sequences(sequences, maxlen=self.max_length)
        ##Turning labels into One-Hot encodings¶
        labels = to_categorical(np.asarray(target))

        word_index = tokenizer_train.word_index
        # Create inverse index mapping numbers to words
        inv_index = {v: k for k, v in tokenizer_train.word_index.items()}


        ##Loading GloVe embeddings
        embeddings_index = {} # We create a dictionary of word -> embedding
        if os.path.isfile('glove.6B.100d.txt'):
            f = open('glove.6B.100d.txt') # Open file
        else:
            print('download it from website')
            response = urllib3.urlopen('https://www.kaggle.com/terenceliu4444/glove6b100dtxt/download')
            data = response.read()
            filename = "glove.6B.100d.txt"
            file_ = open(filename, 'w')
            file_.write(data)
            file_.close()
        # In the dataset, each line represents a new word embedding
        # The line starts with the word and the embedding values follow
        for line in f:
            values = line.split()
            word = values[0] # The first value is the word, the rest are the values of the embedding
            embedding = np.asarray(values[1:], dtype='float32') # Load embedding
            embeddings_index[word] = embedding # Add embedding to our embedding dictionary
        f.close()

        # Create a matrix of all embeddings
        all_embs = np.stack(embeddings_index.values())
        emb_mean = all_embs.mean() # Calculate mean
        emb_std = all_embs.std() # Calculate standard deviation
        word_index = tokenizer_train.word_index
        nb_words = min(self.vocab_size, len(word_index)) # How many words are there actually

        # Create a random matrix with the same mean and std as the embeddings
        embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, self.embedding_dim))

        # The vectors need to be in the same position as their index.
        # Meaning a word with token 1 needs to be in the second row (rows start with zero) and so on

        # Loop over all words in the word index
        for word, i in word_index.items():
            # If we are above the amount of words we want to use we do nothing
            if i >= self.vocab_size:
                continue
            # Get the embedding vector for the word
            embedding_vector = embeddings_index.get(word)
            # If there is an embedding vector, put it in the embedding matrix
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector


        model = Sequential()
        model.add(Embedding(self.vocab_size,
                            self.embedding_dim,
                            input_length=self.max_length,
                            weights = [embedding_matrix],
                            trainable = False))
        model.add(LSTM(128,recurrent_dropout=0.1))
        model.add(Dense(20))
        model.add(Activation('softmax'))
        model.summary()
        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['acc'])
        model.fit(data,labels,validation_split=0.2,epochs=100)

    # def main_test(self):
        ##process testing dataset
        twenty_test = fetch_20newsgroups(subset='test', remove=['headers', 'footers', 'quote'], shuffle=True)
        texts_test= twenty_test.data # Extract text
        targets_test=twenty_test.target# Extract target
        tokenizer_test = Tokenizer(num_words=self.vocab_size) # Setup tokenizer
        tokenizer_test.fit_on_texts(texts_test)
        sequences_test = tokenizer_test.texts_to_sequences(texts_test) # Generate sequences for texting
        data_test=pad_sequences(sequences_test, maxlen=self.max_length)
        ##Turning labels into One-Hot encodings¶
        labels_test=to_categorical(np.asarray(targets_test))
        lstm_predicted= model.predict(data_test)
        print('lstm', lstm_predicted)
        print('label',labels_test)
        print('accuracy',np.mean(np.argmax(lstm_predicted,axis=1) == np.argmax(labels_test,axis=1)))

if __name__ == '__main__':
    LSTM_model()
