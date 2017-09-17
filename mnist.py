from keras.models import Sequential
from keras.layers import Dense
from keras.layers.core import Flatten
from keras.optimizers import SGD
from keras.datasets import mnist
from keras.utils import np_utils

import os
import numpy as np
sample_size = 40
batch_size = 10
nb_classes = 1
nb_epoch = 40

def generate_arrays_from_csv():
    path='/media/zsombi/Data/szakdoga/probafilok'
    files_to_classificate = os.listdir('/media/zsombi/Data/szakdoga/probafilok')
    for f in files_to_classificate:
        if f.find('emg')!=-1:
            emg_file = open(path+'/'+f)
        elif f.find('glove')!=-1:
            glove_file = open(path+'/'+f)
        elif f.find('label')!=-1:
            label_file = open(path+'/'+f)

    emg_string = ''
    glove_string = ''
    label_string=''
    emg_data = np.empty([0])
    glove_data = np.empty([0])
    label_data = np.empty([0])
    train_data = np.empty([0])

    for i in range(40):
        emg_string += emg_file.readline()
        glove_string += glove_file.readline()
    emg_arr = np.fromstring(emg_string, sep=',')
    glove_arr = np.fromstring(glove_string, sep=',')
    print('emgarrshape',np.shape(emg_arr))
    print('glovearrshape',np.shape(glove_arr))


    label_string += label_file.readline()
    label_arr = np.fromstring(label_string,sep=',')
    print(label_arr)




    label_data = np.zeros((1))
    label_data = np_utils.to_categorical(label_data,14)
    emg_data.shape = (10*40)
    glove_data.shape = (22*40)
    train_data = np.concatenate((emg_data, glove_data))
    print('data meret:', np.shape(train_data))
    #print(train_data.shape)
    #print(label_data.shape)
    yield (train_data,label_data)


# Load MNIST dataset
#(X_train, y_train), (X_test, y_test) = mnist.load_data()
#X_train = X_train.reshape(60000, 784)
#X_test = X_test.reshape(10000, 784)
#X_train /= 255
#X_test /= 255
#Y_Train = np_utils.to_categorical(y_train, nb_classes)
#Y_Test = np_utils.to_categorical(y_test, nb_classes)

# Logistic regression model
model = Sequential()
model.add(Dense(output_dim=100, input_shape=(batch_size,1,1280), init='normal'))
model.add(Flatten())
model.add(Dense(output_dim=14,activation='softmax'))
model.compile(optimizer=SGD(lr=0.05), loss='categorical_crossentropy', metrics=['accuracy'])


# Train
history = model.fit_generator(generate_arrays_from_csv(),nb_epoch=1,samples_per_epoch=10, verbose=1,show_accuracy=True)

# Evaluate
#evaluation = model.evaluate(X_test, Y_Test, verbose=1)
#print('Summary: Loss over the test dataset: %.2f, Accuracy: %.2f' % (evaluation[0], evaluation[1]))
generate_arrays_from_csv()