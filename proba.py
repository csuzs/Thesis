import os
rootdir='/media/zsombi/Data/szakdoga/probafilok'
import csv
import scipy.io as sio
import numpy as np
from keras.utils import np_utils
from keras.layers import *
from keras.models import Sequential
from keras.optimizers import *

E1_num_of_classes = 12
E2_num_of_classes = 17
E3_num_if_classes = 23
total_number_of_classes = 52 + 1 ## doing nothing is class 0


batch_size = 1
def beolvasoproba():
    onefile = '/media/zsombi/Data/szakdoga/probafilok/  S1_A1_E1emgcsv'

    # f = open(onefile)
    # str = ''
    # tmp = ''



    files_to_classificate = os.listdir('/media/zsombi/Data/szakdoga/probafilok')
    print(files_to_classificate)
    print(files_to_classificate[1].find('lelci'))
    print('/media/zsombi/Data/szakdoga/probafilok' + '/' + files_to_classificate[1])
    f = open('/media/zsombi/Data/szakdoga/probafilok' + '/' + files_to_classificate[1])
    print(f.readline())

    # with open(onefile, "rb") as csvFile:
    #    for row in csvFile:
    #        print(row)


    path = '/media/zsombi/Data/szakdoga/probafilok'
    files_to_classificate = os.listdir(path)
    for f in files_to_classificate:
        if f.find('emg') != -1:
            emg_file = open(path + '/' + f)
        elif f.find('glove') != -1:
            glove_file = open(path + '/' + f)
        elif f.find('label') != -1:
            label_file = open(path + '/' + f)
    emg_string = ''
    glove_string = ''
    emg_data = np.empty([0])
    glove_data = np.empty([0])
    for i in range(40):

        glove_string =glove_file.readline()

        emg_arr = np.fromstring(emg_file.readline(), sep=',')
        glove_arr = np.fromstring(glove_file.readline(), sep=',')
        emg_data = np.concatenate((emg_data, emg_arr))
        glove_data = np.concatenate((glove_data, glove_arr))

    label_data = label_file.readline()
    label_data = np.zeros((1))
    label_data[0] = int(label_data)

    emg_data.shape = (400)
    glove_data.shape = (22*40)
    train_data = np.concatenate((emg_data,glove_data))
    print(train_data.shape)
    print(label_data.shape)


def egyadatsorralefutohalotanitas():

    path = '/media/zsombi/Data/szakdoga/probafilok'
    files_to_classificate = os.listdir(path)
    for f in files_to_classificate:
        if f.find('emg') != -1:
            emg_file = open(path + '/' + f)
        elif f.find('glove') != -1:
            glove_file = open(path + '/' + f)
        elif f.find('label') != -1:
            label_file = open(path + '/' + f)

    emg_string = ''
    glove_string = ''
    label_string = ''
    emg_data = np.empty([0])
    glove_data = np.empty([0])
    label_data = np.empty([0])
    train_data = np.empty([0])

    for i in range(40):
        emg_arr = np.fromstring(emg_file.readline(), sep=',')
        glove_arr = np.fromstring(glove_file.readline(), sep=',')
        emg_data = np.concatenate((emg_data, emg_arr))
        glove_data = np.concatenate((glove_data, glove_arr))


    label_string += label_file.readline()
    label_arr = np.fromstring(label_string, sep=',')
    print(emg_data.shape)
    print(glove_data.shape)

    label_data = np.zeros((1))
    label_data = np_utils.to_categorical(label_data, 14)
    emg_data.shape = (10 * 40)
    glove_data.shape = (22 * 40)
    train_data = np.concatenate((emg_data, glove_data))
    train_data.shape = (1,1280)
    print('data meret:', np.shape(train_data))
    print('label meret:',np.shape(label_data))
    # print(train_data.shape)
    # print(label_data.shape)
    model = Sequential()
    model.add(Dense(output_dim=100, input_shape=(1280,), init='normal'))
    model.add(Dense(output_dim=14, activation='softmax'))
    model.compile(optimizer=SGD(lr=0.05), loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(train_data, label_data, nb_epoch=10, batch_size=1, verbose=1)


def regiadattalmukodott():
    ##################
    model = Sequential()
    model.add(Dense(output_dim=100, input_shape=(1280,), init='normal'))
    model.add(Dense(output_dim=14, activation='softmax'))
    model.compile(optimizer=SGD(lr=0.05), loss='categorical_crossentropy', metrics=['accuracy'])



    #################x
    path = '/media/zsombi/Data/szakdoga/probafilok'
    files_to_classificate = os.listdir(path)
    for f in files_to_classificate:
        if f.find('emg') != -1:
            emg_file = open(path + '/' + f)
        elif f.find('glove') != -1:
            glove_file = open(path + '/' + f)
        elif f.find('label') != -1:
            label_file = open(path + '/' + f)

    eof = False
    while eof == False:

        emg_string = ''
        glove_string = ''
        label_string = ''


        train_data_batch = np.empty((10,1280))
        label_data_batch = np.empty((10,14))
        for i in range(10):
            train_data = np.empty((1, 1280))
            emg_data = np.empty([0])
            glove_data = np.empty([0])
            label_data = np.empty([0])
            for j in range(40):
                emg_arr = np.fromstring(emg_file.readline(), sep=',')
                glove_arr = np.fromstring(glove_file.readline(), sep=',')
                emg_data = np.concatenate((emg_data, emg_arr))
                glove_data = np.concatenate((glove_data, glove_arr))


            label_string = label_file.readline()
            if ""==label_string:
                eof = True

            label_arr = np.fromstring(label_string, sep=',')
            #print(emg_data.shape)
            #print(glove_data.shape)

            label_data = np_utils.to_categorical(label_arr[0], 14)
            label_data_batch[i] = label_data
            emg_data.shape = (10 * 40)
            glove_data.shape = (22 * 40)
            train_data = np.concatenate((emg_data, glove_data))
            train_data.shape = (1,1280)
            train_data_batch[i]=train_data
        history = model.fit(train_data_batch, label_data_batch, nb_epoch=1, batch_size=10, verbose=1)
        # print(train_data.shape)
        # print(label_data.shape)


if __name__=="__main__":

        train_path_array = ["/media/zsombi/Data/szakdoga/DB1/trainS19_A1_E1_windowed.csv",
                        "/media/zsombi/Data/szakdoga/DB1/trainS19_A1_E2_windowed.csv",
                        "/media/zsombi/Data/szakdoga/DB1/trainS19_A1_E3_windowed.csv"]
        test_path_array = ['/media/zsombi/Data/szakdoga/DB1/test/trainS9_A1_E3_windowed.csv',
                       '/media/zsombi/Data/szakdoga/DB1/test/trainS9_A1_E1_windowed.csv',
                       '/media/zsombi/Data/szakdoga/DB1/test/trainS9_A1_E2_windowed.csv']

        model = Sequential()
        model.add(Dense(output_dim=4000, input_shape=(1280,), init='normal'))
        model.add(Dense(output_dim=1000,init='normal'))
        model.add(Dense(output_dim=500, init='normal'))
        model.add(Dense(output_dim=14,init ='normal',activation='softmax'))
        model.compile(optimizer=SGD(lr=0.05), loss='categorical_crossentropy', metrics=['accuracy'])


        path = '/media/zsombi/Data/szakdoga/probafilok'
        files_to_classificate = os.listdir(path)
        for f in files_to_classificate:
            if f.find('windowed') != -1:
                windowed_datas = open(path + '/' + f)

        train_data_batch = np.empty((batch_size, 1280))
        label_data_batch = np.empty((batch_size, 14))

        for i in range(batch_size):
            emg_string = ''
            glove_string = ''
            label_string = ''
            emg_data = np.empty([0])
            glove_data = np.empty([0])
            label_data = np.empty([0])
            train_data = np.empty([0])

            label_string += windowed_datas.readline()
            label_arr = np.fromstring(label_string, sep=',')
            label_data = np.zeros((1),dtype='float64')
            label_data = np_utils.to_categorical(label_data, 14)
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
        # print(train_data.shape)
        # print(label_data.shape)


        history = model.fit(train_data_batch, label_data_batch, nb_epoch=10, batch_size=batch_size, verbose=1)

        train_data_batch = np.empty((batch_size, 1280))
        label_data_batch = np.empty((batch_size, 14))

        for i in range(batch_size):

            emg_string = ''
            glove_string = ''
            label_string = ''
            emg_data = np.empty([0])
            glove_data = np.empty([0])
            label_data = np.empty([0])
            train_data = np.empty([0])

            label_string += windowed_datas.readline()
            label_arr = np.fromstring(label_string, sep=',')
            label_data = np.zeros((1), dtype='float64')
            label_data = np_utils.to_categorical(label_data, 14)
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

        evaluation = model.evaluate(train_data_batch, label_data_batch, batch_size=batch_size, verbose=1)
        print('Summary: Loss over the test dataset: %.2f, Accuracy: %.2f' % (evaluation[0], evaluation[1]))