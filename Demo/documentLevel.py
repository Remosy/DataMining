import keras
import tensorflow as tf
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Activation,Flatten
from keras.layers import Conv1D, Embedding,GlobalMaxPooling1D,Dropout,MaxPooling1D
from keras import regularizers, optimizers
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve,roc_auc_score
import matplotlib.pyplot as plt
import numpy as np

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
        train_data = np.array(self.train_data)
        labels = np.array(self.labels)
        # Train Set Size: 75%
        # Test Set Size: 15%
        X, X_test, y, y_test = train_test_split(train_data, labels, train_size= 0.90,test_size=0.10,random_state=43)
        # Validate Set Size: 15%
        X_train, X_v, y_train, y_v = train_test_split(X, y, train_size=0.88888888888, test_size=0.11111111111,random_state=43)
        print("Train data has: "+str(len(X_train)))
        print("Test data has: "+str(len(X_test)))
        model = Sequential()
        model.add(Conv1D(256, 5, activation='relu', padding='same', input_shape=(10, 100)))
        model.add(GlobalMaxPooling1D())
        model.add(Dropout(0.5)) # fraction rate for input, prevent over fitting by setting 0.5
        model.add(Dense(256,kernel_regularizer=regularizers.l2(0.0001), activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        #model.fit(X_train, y_train, batch_size=20, epochs=200, validation_data=(X_test,y_test), shuffle=True)
        model.fit(X_train, y_train, batch_size=20, epochs=1, validation_data=(X_v, y_v), shuffle=True)
        sgd = optimizers.SGD(lr=0.02)
        model.compile(optimizer=sgd,
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        model.fit(X_train, y_train, batch_size=100, epochs=200, validation_data=(X_v, y_v))
        test_prediction = model.predict(X_test)

        #y_test = np.array(y_test)
        #y_test = np.transpose(y_test)
        #test_prediction = np.array(test_prediction)

        fpr, tpr ,th= roc_curve(y_test,test_prediction, pos_label=1)
        auc_score = roc_auc_score(y_test,test_prediction)
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange',
                 lw=2, label='ROC curve (area = %0.2f)' % auc_score)
        plt.plot([0,1],[0,1], color='navy',lw=2, linestyle='--')
        plt.xlim([0.0,1.0])
        plt.ylim([0.0,1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.show()