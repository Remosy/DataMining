import keras
import tensorflow as tf
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Activation,Flatten
from keras.layers import Conv1D, Embedding,GlobalMaxPooling1D,Dropout,MaxPooling1D
from sklearn.model_selection import train_test_split

class testCNN():
    def __init__(self):
            super().__init__()
            train_data = [[[-0.06240204, -0.09176608, -0.03593158, -0.0451744 , -0.03068076],
            [ 0.06228806,  0.05816831,  0.01400619,  0.09236281,  0.02532971],
            [-0.07197677,  0.04217994, -0.0074635 ,  0.05444086, -0.01719033],
            [-0.04877526, -0.06917289,  0.08718955,  0.02770192,  0.0810696 ],
            [ 0.05192148, -0.04228239, -0.04150534,  0.00501796, -0.01101556]],
            [[ 0.03411563,  0.01194958,  0.07072639,  0.00889047,  0.02655569],
            [ 0.02925151, -0.09861016, -0.08716575,  0.06121465,  0.06666205],
            [-0.0580556 , -0.05903967,  0.09896692, -0.07431905,  0.06224633],
            [ 0.04030856, -0.03498285,  0.09061892,  0.05576834, -0.08827306]],
            [-0.09440739,  0.0190507 ,  0.0151092 ,  0.08919219,  0.03724141]]

            labels = [1,1,0]

            X_train, X_test, y_train, y_test = train_test_split(train_data, labels, test_size=0.33, random_state=42)
            print("Train data has: "+str(len(X_train)))
            print("Test data has: "+str(len(X_test)))
            sess = tf.Session()
            K.set_session(sess)
            model = Sequential()
            model.add(Conv1D(64, 3, activation='relu', input_shape=(10, 5)))
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

if __name__ == '__main__':
    xxx = testCNN()