from keras.models import Sequential

from keras.layers.core import *
from keras.optimizers import SGD
from keras.utils import np_utils
import glob
import os
import numpy as np
num_of_classes = 52+1
num_of_classes_without_null = 52
sample_size = 40
batch_size = 32
nb_classes = 1
nb_epoch = 40
evalsize = 1000
train_path_array = ["/media/zsombi/Data/szakdoga/DB1/trainS19_A1_E1_windowed.csv",
                    "/media/zsombi/Data/szakdoga/DB1/trainS19_A1_E2_windowed.csv",
                    "/media/zsombi/Data/szakdoga/DB1/trainS19_A1_E3_windowed.csv"]
test_path_array = ['/media/zsombi/Data/szakdoga/DB1/test/trainS9_A1_E3_windowed.csv',
                   '/media/zsombi/Data/szakdoga/DB1/test/trainS9_A1_E1_windowed.csv',
                   '/media/zsombi/Data/szakdoga/DB1/test/trainS9_A1_E2_windowed.csv']
def Feed_files_into_network(model):
    Trainfilepaths = []
    Testfilepaths =  []
    for subdir, dirs, files in os.walk('/media/zsombi/Data/szakdoga/DB1/proba/'):
        for file in files:
            Trainfilepaths.append(subdir + os.sep + file)
    for subdir, dirs, files in os.walk('/media/zsombi/Data/szakdoga/DB1/probatest/'):
        for file in files:
            Testfilepaths.append(subdir + os.sep + file)

    for path in Trainfilepaths:
        Train_network(model,path)
    for path in Testfilepaths:
        evaluate_trained_network(model,path)
def Train_network(model,path):


    windowed_datas = open(path)

    while 1:

        train_data_batch = np.empty((batch_size, 1280))
        label_data_batch = np.empty((batch_size, num_of_classes))


        for i in range(batch_size):
            emg_string = ''
            glove_string = ''
            label_string = ''
            emg_data = np.empty([0])
            glove_data = np.empty([0])
            label_data = np.empty([0])
            train_data = np.empty([0])
            label_string = windowed_datas.readline()
            if label_string == '':
                return
            label_arr = np.fromstring(label_string, sep=',')
            #label_data = np.zeros(label_arr, dtype='float64')
            label_data = np_utils.to_categorical(label_arr[0], num_of_classes)
            for j in range(40):
                emg_arr = np.fromstring(windowed_datas.readline(), sep=',')
                emg_data = np.concatenate((emg_data, emg_arr))
            for j in range(40):
                glove_arr = np.fromstring(windowed_datas.readline(), sep=',')
                glove_data = np.concatenate((glove_data, glove_arr))

            emg_data.shape = (10 * 40)
            glove_data.shape = (22 * 40)
            train_data = np.concatenate((emg_data, glove_data))
            train_data.shape = (1, 1280)
            train_data_batch[i] = train_data
            label_data_batch[i] = label_data


        history = model.fit(train_data_batch, label_data_batch, nb_epoch=2, batch_size=batch_size, verbose=1,show_accuracy=True)

def evaluate_trained_network(model,path):
    windowed_datas = open(path)


    while 1:

        train_data_batch = np.empty((evalsize, 1280))
        label_data_batch = np.empty((evalsize, num_of_classes))

        for i in range(evalsize):
            emg_string = ''
            glove_string = ''
            label_string = ''
            emg_data = np.empty([0])
            glove_data = np.empty([0])
            label_data = np.empty([0])
            train_data = np.empty([0])
            label_string = windowed_datas.readline()
            if label_string == '':
                return
            label_arr = np.fromstring(label_string, sep=',')
            # label_data = np.zeros(label_arr, dtype='float64')
            label_data = np_utils.to_categorical(label_arr[0], num_of_classes)
            for j in range(40):
                emg_arr = np.fromstring(windowed_datas.readline(), sep=',')
                emg_data = np.concatenate((emg_data, emg_arr))
            for j in range(40):
                glove_arr = np.fromstring(windowed_datas.readline(), sep=',')
                glove_data = np.concatenate((glove_data, glove_arr))

            emg_data.shape = (10 * 40)
            glove_data.shape = (22 * 40)
            train_data = np.concatenate((emg_data, glove_data))
            train_data.shape = (1, 1280)
            train_data_batch[i] = train_data
            label_data_batch[i] = label_data
        evaluation = model.evaluate(train_data_batch, label_data_batch, verbose=1)
        print('Summary: Loss over the test dataset: %.2f, Accuracy: %.2f' % (evaluation[0], evaluation[1]))
# Evaluate

if __name__=='__main__':
    model = Sequential()
    model.add(Dense(output_dim=2000, input_shape=(1280,), init='normal'))
    model.add(Dense(output_dim =1000))

    model.add(Dropout(0.4))
    model.add(Dense(output_dim=500))
    model.add(Dense(output_dim=200))


    model.add(Dense(output_dim=num_of_classes, init='normal', activation='softmax'))
    model.compile(optimizer=SGD(lr=0.05), loss='categorical_crossentropy', metrics=['accuracy'])
    Feed_files_into_network(model)