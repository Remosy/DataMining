import keras
import tensorflow as tf
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Activation,Flatten
from keras.layers import Conv1D, Embedding,GlobalMaxPooling1D,Dropout,MaxPooling1D
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
        X_train, X_test, y_train, y_test = train_test_split(self.train_data, self.labels, test_size=0.33, random_state=42)
        print("Train data has: "+str(len(X_train)))
        print("Test data has: "+str(len(X_test)))
        sess = tf.Session()
        K.set_session(sess)
        model = Sequential()
        model.add(Conv1D(64, 3, activation='relu', input_shape=(20, 100)))
        model.add(MaxPooling1D(3))
        model.add(Conv1D(128, 3, activation='relu'))
        model.add(GlobalMaxPooling1D())
        model.add(Dropout(0.5)) # fraction rate for input, prevent over fitting by setting 0.5
        model.add(Dense(1, activation='sigmoid'))

        model.compile(optimizer='rmsprop',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        model.fit(X_train, y_train, batch_size=16, epochs=10)
        score = model.evaluate(X_test, y_test, batch_size=16)