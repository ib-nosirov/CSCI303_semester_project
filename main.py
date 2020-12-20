import preprocess
from preprocess import Process
import sklearn
from sklearn import neural_network 
import pandas as pd
import numpy as np
import tensorflow as tf



model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(2, 79)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(96, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid'),
])
model.compile(optimizer='adam',
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=['accuracy'])

score = np.empty([1,30])

for i in range(30):
    data = pd.read_excel('time_series_375_prerpocess_en.xlsx')
    obj = Process()
    data = obj.process(data)

    Xtrain, Ytrain, Xtest, Ytest = obj.train_test_split(data)
    Xtrain = obj.scale(Xtrain)
    Xtest = obj.scale(Xtest)
    model.fit(Xtrain, Ytrain, epochs=100)
    test_loss, test_acc = model.evaluate(Xtest,  Ytest, verbose=2)
    score[0][i] = test_acc

print('\nMean test accuracy:', np.mean(score))
print('\nStandard Deviation test accuracy:', np.std(score))
