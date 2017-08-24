

#dataglove- 25 Hz sample rate
#Electrodes: 100 Hz
#from keras.models import Sequential
#from keras.layers import Dense, Activation
import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as mpl
import _pickle as pickle
import sys
ROOT_DIRECTION = '/home/zsombi/szakdoga/ninapro_classif/DB1/s2'
TO_SAVE_DIRECTION = '/media/zsombi/Data/szakdoga/DB1/windowed_datas'
SAMPLING_RATE = 100 #Hz
WINDOW_SIZE_IN_SAMPLES = 40 # 400 ms
SLIDING_SIZE = 1   # 10 ms

def load_data_into_pickles():
    for subdir, dirs, files in os.walk(ROOT_DIRECTION):
        for file in files:
            # print os.path.join(subdir, file)
            filepath = subdir + os.sep + file

            if filepath.endswith(".mat"):
                print('kezodik a feldolgozas: ')
                print(filepath)
                data=load_data(filepath)
                print('adat beolvasva! ')
                normalize_data(data)
                print('adat normalizalva')
                windowed_datas = get_windows(data)
                print('ablakolas kesz: ')
                print(filepath)
                filename = file.replace('mat','p')
                fullfiledir = TO_SAVE_DIRECTION + filename
                pickle.dump(windowed_datas,open(fullfiledir,'wb'))
                print('kiirodott: ')
                print(filepath)


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
    divider = max_element-min_element
    if (max_element-min_element)<eps:
            divider = eps
    normalized = (x-min_element)/divider
    return normalized

def normalize_data(data):
    for i in range(10):
        data["emg"][:, i] = norm(data["emg"][:, i])
    for i in range(22):
        data["glove"][:, i] = norm(data["glove"][:, i])
    if data.__contains__('inclin'):
        for i in range(2):
            data['inclin'][:,i]=norm(data['inclin'][:,i])
    if data.__contains__('acc'):
        for i in range(36):
            data['acc'][:,i]=norm(data['acc'][:,i])
def setLabel(labels):

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

        return np.zeros([1],dtype=np.int8)



def get_windows(data):
    if not str(data.keys()).__contains__('inclin'):
        data_windows={'glove':[], 'emg':[], 'restimulus':[],'label':[]}
    else:
        data_windows = {'glove': [], 'emg': [],'acc':[],'inclin':[], 'restimulus': [], 'label': []}

    glovedata=data['glove']
    emgdata=data['emg']
    restimulus=data['restimulus']


    i=0
    j=0

    while j + WINDOW_SIZE_IN_SAMPLES < len(restimulus):
        #windows.append(glovedata[j:j + WINDOW_SIZE_IN_SAMPLES])
        data_windows['glove'].append(glovedata[j:j+WINDOW_SIZE_IN_SAMPLES])
        data_windows['emg'].append(emgdata[j:j+WINDOW_SIZE_IN_SAMPLES])

        if(data.__contains__('acc')):                                                         #acc and inclin only occurs in DB2 and DB3
            data_windows['acc'].append(data['acc'][j:j+WINDOW_SIZE_IN_SAMPLES ])
        if(data.__contains__('inclin')):
            data_windows['inclin'].append(data['inclin'][j:j+WINDOW_SIZE_IN_SAMPLES])

        data_windows['restimulus'].append(restimulus[j:j+WINDOW_SIZE_IN_SAMPLES])
        data_windows['label'].append(setLabel(restimulus[j:j+WINDOW_SIZE_IN_SAMPLES]))


        i = i + 1
        j = j + SLIDING_SIZE
    print(np.shape(data_windows['glove']))
    print(np.shape(data_windows['emg']))
    print(np.shape(data_windows['restimulus']))
    print(np.shape(data_windows['label']))
    print(data_windows['label'])
    return data_windows









if  __name__=="__main__":

    #data = load_data('/home/zsombi/szakdoga/ninapro_classif/DB1/s1/S1_A1_E3.mat')
    #normalize_data(data)
    #windowed_datas = get_windows(data)
    #print(np.shape(windowed_datas['acc']))
    load_data_into_pickles()
    #data = load_data('/home/zsombi/szakdoga/ninapro_classif/DB1/s2/S2_A1_E2.mat')
    #pickle.dump(data,open('lelci.p','wb'))
    #d = pickle.load(open('lelci.p','rb'))

    #normalize_data(data)
    #windowed_datas = get_windows(data)
    #mpl.plot(data['restimulus'])
    #mpl.plot(data['emg'])
    #mpl.plot(data["glove"])
    #mpl.show()



#for i in data.values():
#	if type(i) == np.ndarray:
#		print(i)
#		print(i.shape)
#model = Sequential()
#model.add(Dense(500,input_dim = 784,activation = 'sigmoid'))
#model.add(Dense(10,input_dim = 500,activation = 'sigmoid'))
#model.compile(optimizer='rmsprop',loss = 'binary_crossentropy',metrics = ['accuracy'])


# generate dummy data

#data = np.random.random((1000, 784))
#labels = np.random.randint(2, size=(1000, 10))

# train the model, iterating on the data in batches
# of 32 samples
#model.fit(data, labels, nb_epoch=100, batch_size=10)