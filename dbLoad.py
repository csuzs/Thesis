# dataglove- 25 Hz sample rate
# Electrodes: 100 Hz
# from keras.models import Sequential
# from keras.layers import Dense, Activation
import os
import sys
import csv
import random as rand
import numpy as np
import scipy.io as sio
import _pickle as pickle
import matplotlib.pyplot as mpl
ROOT_DIRECTION = '/home/zsombi/szakdoga/ninapro_classif/DB1'
TO_SAVE_DIRECTION = '/media/zsombi/Data/szakdoga/probafilok/  '
SAMPLING_RATE = 100  # Hz
WINDOW_SIZE_IN_SAMPLES = 40  # 400 ms
SLIDING_SIZE = 1  # 10 ms
E1_num_of_classes = 12
E2_num_of_classes = 17
E3_num_if_classes = 23
def load_data_into_pickles():
    for subdir, dirs, files in os.walk(ROOT_DIRECTION):
        for file in files:
            # print os.path.join(subdir, file)
            filepath = subdir + os.sep + file

            if filepath.endswith(".mat"):
                print('kezodik a feldolgozas:')
                print(filepath)
                data = load_data(filepath)
                print('adat beolvasva! ')
                normalize_data(data)
                print('adat normalizalva')
                windowed_datas = get_windows(data)
                print('ablakolas kesz: ')
                print(filepath)
                filename = file.replace('mat', 'p')
                fullfiledir = TO_SAVE_DIRECTION + filename
                pickle.dump(windowed_datas, open(fullfiledir, 'wb'))
                print('kiirodott: ')
                print(filepath)


def save_data_into_csv(root):
    for subdir, dirs, files in os.walk(root):
        for file in files:
            # print os.path.join(subdir, file)
            filepath = subdir + os.sep + file

            if filepath.endswith(".mat"):
                print('kezodik a feldolgozas:')
                print(filepath)
                data = load_data(filepath)
                print('adat beolvasva! ')
                normalize_data(data)
                print('adat normalizalva')
                windowed_datas = get_windows(data)
                print('ablakolas kesz: ')
                print(filepath)
                order = np.arange(len(windowed_datas['glove'][:]))

                rand.shuffle(order)

                filename = file.replace('mat', 'csv')
                fullfiledir = TO_SAVE_DIRECTION + filename
                if windowed_datas.__contains__('inclin') & windowed_datas.__contains__('acc'):
                    with open(fullfiledir, 'w') as csvfile:
                        writer = csv.writer(csvfile, delimiter=',')
                        for i in order:
                            writer.writerow(windowed_datas['label'][i])
                            writer.writerow(windowed_datas['glove'][i])
                            writer.writerow(windowed_datas['emg'][i])
                            writer.writerow(windowed_datas['acc'][i])

                        csvfile.close()
                else:
                    labels = np.asarray(windowed_datas['label'])[order]
                    glove  = np.asarray(windowed_datas['glove'])[order]
                    emg = np.asarray(windowed_datas['emg'])[order]
                    print(np.shape(labels))
                    print(np.shape(glove))
                    print(np.shape(emg))
                    with open(fullfiledir.replace('.','_windowed_preprocessed_data.'),'wb') as f:
                        for i in range(len(labels)):

                            np.savetxt(f,labels[i],delimiter=',',fmt = '%1.0i')
                            np.savetxt(f,emg[i][:][:],delimiter=',',fmt = '%1.7f')
                            np.savetxt(f,glove[i][:][:],delimiter=',',fmt='%1.7f')







                        #np.savetxt(fullfiledir.replace('.','label.'),labels,fmt='%.0i')
                        #with open(fullfiledir.replace('.','glove.'), 'wb') as f:
                        #    for a in windowed_datas['glove'][:]:
                        #        np.savetxt(f,a,delimiter=',',fmt='%1.8f')
                        #
                        #with open(fullfiledir.replace('.', 'emg'), 'wb') as f:
                        #    for a in windowed_datas['emg'][:]:
                        #        np.savetxt(f,a,delimiter=',',fmt='%1.8f')



def load_data(dir):
    data = sio.loadmat(dir)
    print('keys = ' + str(data.keys()))
    for i in data.keys():
        print(i)
        if type(data[i]) == np.ndarray:
            print(np.shape(data[i]))

        else:
            print(data[i])
        if (i == "exercise"):
            print(data[i][0])
    return data


def norm(x):
    max_element = np.amax(x)
    min_element = np.amin(x)
    eps = sys.float_info.epsilon
    divider = max_element - min_element
    if (max_element - min_element) < eps:
        divider = eps
    normalized = (x - min_element) / divider
    return normalized


def normalize_data(data):
    for i in range(10):
        data["emg"][:, i] = norm(data["emg"][:, i])
    for i in range(22):
        data["glove"][:, i] = norm(data["glove"][:, i])
    if data.__contains__('inclin'):
        for i in range(2):
            data['inclin'][:, i] = norm(data['inclin'][:, i])
    if data.__contains__('acc'):
        for i in range(36):
            data['acc'][:, i] = norm(data['acc'][:, i])


def setlabel(labels):
    label_num = 0
    label_count = 0
    null_count = 0
    for i in labels:

        if i.any() != 0:
            label_num = i
            label_count += 1
        else:
            null_count += 1

    if label_count > null_count:

        return label_num
    else:

        return np.zeros([1], dtype=np.int8)


def get_windows(data):
    if not str(data.keys()).__contains__('inclin'):
        # data_windows={'glove':[], 'emg':[], 'restimulus':[],'label':[]}
        data_windows = {'glove': [], 'emg': [], 'label': []}
    else:

        # data_windows = {'glove': [], 'emg': [],'acc':[],'inclin':[], 'restimulus': [], 'label': []}
        data_windows = {'glove': [], 'emg': [], 'acc': [], 'inclin': [], 'label': []}

    i = 0
    j = 0
    datasize = len(data['restimulus'])
    while j + WINDOW_SIZE_IN_SAMPLES < datasize:
        # windows.append(glovedata[j:j + WINDOW_SIZE_IN_SAMPLES])
        data_windows['glove'].append(data['glove'][j:j + WINDOW_SIZE_IN_SAMPLES])
        data_windows['emg'].append(data['emg'][j:j + WINDOW_SIZE_IN_SAMPLES])

        if (data.__contains__('acc')):  # acc and inclin only occurs in DB2 and DB3
            data_windows['acc'].append(data['acc'][j:j + WINDOW_SIZE_IN_SAMPLES])
        if (data.__contains__('inclin')):
            data_windows['inclin'].append(data['inclin'][j:j + WINDOW_SIZE_IN_SAMPLES])

        # data_windows['restimulus'].append(restimulus[j:j+WINDOW_SIZE_IN_SAMPLES])
        data_windows['label'].append(setlabel(data['restimulus'][j:j + WINDOW_SIZE_IN_SAMPLES]))

        i = i + 1
        j = j + SLIDING_SIZE
    print('glove:')
    print(np.shape(data_windows['glove']))
    print(np.shape(data_windows['glove'][0]))
    print(len(data_windows['glove'][:]))
    print('emg:')
    print(np.shape(data_windows['emg']))
    # print('restimulus:')
    # print(np.shape(data_windows['restimulus']))
    print('label:')
    print(np.shape(data_windows['label'])
          )
    return data_windows


if __name__ == "__main__":
    #save_data_into_csv('/home/zsombi/szakdoga/ninapro_classif/sample')

    data = load_data('/home/zsombi/szakdoga/ninapro_classif/DB1/s1/S1_A1_E1.mat')
    print('\n\n\n')
    data2 = load_data('/home/zsombi/szakdoga/ninapro_classif/DB1/s1/S1_A1_E2.mat')
    print('\n\n\n')
    data3 = load_data('/home/zsombi/szakdoga/ninapro_classif/DB1/s1/S1_A1_E3.mat')
    print('\n\n\n')
    print('data3 minimuma:',min(data3["restimulus"]))
    print('data3 maximuma:',max(data3["restimulus"]))
    print('data1 minimuma:',min(data["restimulus"]))
    print('data1 maximuma:',max(data["restimulus"]))
    print('data2 minimuma:',min(data2["restimulus"]))
    print('data2 maximuma:',max(data2["restimulus"]))
    # print(windowed_datas.keys())
    # print('\n',type(windowed_datas))
    # print(np.shape(windowed_datas['acc']))
    # load_data_into_pickles()
    # data = load_data('/home/zsombi/szakdoga/ninapro_classif/DB1/s2/S2_A1_E2.mat')
    # pickle.dump(data,open('lelci.p','wb'))
    # d = pickle.load(open('lelci.p','rb'))

    normalize_data(data3)
    #windowed_datas = get_windows(data)
    mpl.plot(data['restimulus'])
    mpl.plot(data2['restimulus'])
    mpl.plot(data3['restimulus'])
    mpl.show()