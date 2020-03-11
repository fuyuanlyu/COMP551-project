from __future__ import print_function

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.datasets import imdb
from keras.layers.convolutional import  Conv1D
from keras.layers.convolutional import MaxPooling1D

def LSTM_IMDB():
    max_features = 20000##5000
    # cut texts after this number of words (among top max_features most common words)
    maxlen = 500
    batch_size = 64

    print('Loading data...')
    # (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
    import numpy as np
    # save np.load
    np_load_old = np.load
    # modify the default parameters of np.load
    np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
    # call load_data with allow_pickle implicitly set to true
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000)
    # restore np.load for future normal usage
    np.load = np_load_old
    print(len(x_train), 'train sequences')
    print(len(x_test), 'test sequences')

    print('Pad sequences (samples x time)')
    x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
    x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
    print('x_train shape:', x_train.shape)
    print('x_test shape:', x_test.shape)

    print('Build model...')
    model = Sequential()
    model.add(Embedding(max_features, 128,input_length=maxlen))
    model.add(Conv1D(filters=32,kernel_size=3,padding='same',activation='relu'))
    model.add(Conv1D(filters=16,kernel_size=3,padding='same',activation='relu'))
    model.add(MaxPooling1D(pool_size=4))

    model.add(LSTM(128,dropout=0.1))
    # model.add(LSTM(128))
    model.add(Dense(1, activation='sigmoid')) #sigmoid=0.82

    # try using different optimizers and different optimizer configs
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    print('Train...')
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=3,
              validation_data=(x_test, y_test))
    score, acc = model.evaluate(x_test, y_test,
                                batch_size=batch_size)
    print('Test score:', score)
    print('Test accuracy:', acc)

if __name__ == '__main__':
    LSTM_IMDB()