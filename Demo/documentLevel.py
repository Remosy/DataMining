import keras
import tensorflow as tf
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Activation,Flatten
from keras.layers import Conv1D, Embedding,GlobalMaxPooling1D,Dropout,MaxPooling1D
from keras import regularizers, optimizers
from sklearn.model_selection import train_test_split

from preprocess import PreProcess
from sentenceLevel import SentenceLevel

class DocumentLevel:
    def __init__(self, data_vectors, labels):
        self.train_data = data_vectors
        self.labels = labels
        self.train()

    def getWordMatrix(self):
        return

    def getDocInput(self):
        return

    def train(self):
        X_train, X_test, y_train, y_test = train_test_split(self.train_data, self.labels, test_size=0.15,random_state=987)
        print("Train data has: "+str(len(X_train)))
        print("Test data has: "+str(len(X_test)))
        model = Sequential()
        model.add(Conv1D(64, 5, activation='relu', padding='same', input_shape=(45, 200),kernel_initializer='random_uniform'))
        model.add(MaxPooling1D(2))
        model.add(Conv1D(128, 2, activation='relu',kernel_initializer='random_uniform'))
        model.add(MaxPooling1D())
        model.add(Flatten())
        model.add(Dropout(0.5)) # fraction rate for input, prevent over fitting by setting 0.5
        model.add(Dense(80,kernel_regularizer=regularizers.l2(0.000),
                  activation='sigmoid',
                        kernel_initializer='random_uniform'))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        model.fit(X_train, y_train, batch_size=200, epochs=200, validation_data=(X_test,y_test), shuffle=True)
        sgd = optimizers.SGD(lr=0.02)
        model.compile(optimizer=sgd,
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        model.fit(X_train, y_train, batch_size=100, epochs=200, validation_data=(X_test, y_test))
        score = model.evaluate(X_test, y_test)