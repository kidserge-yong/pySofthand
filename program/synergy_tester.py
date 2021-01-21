import os
import sys
import csv
import torch
import numpy as np

try:
    sys.path.append('../module/')
    from qbrobot import *
    from smk import *
    from synergy import *
    from menu import *
    from cnn import *
except ImportError:
    raise RuntimeError('cannot import module, make sure sys.path is correct')

inv = lambda n: np.linalg.inv(n)
tp = lambda n: np.transpose(n)
dot = lambda x, y: np.dot(x,y)

def plot(graph):
    plt.plot(graph)
    plt.show() 

def multiplot(graph, channel_dir = 1):
    if channel_dir > 0:
        graph = tp(graph)
    fig, axs = plt.subplots(len(graph), 1)
    for i, item in enumerate(graph):
        axs[i].plot(item)
    plt.show()

def array_normalize(data, array_min=None, array_max=None):
    """
    data[[time] x channel]: data of each channel and time
    """
    ans = []
    if not array_max == None:
        max_ans = array_max
    else:
        max_ans = [max(channel) for channel in data]
    if not array_min == None:
        min_ans = array_min
    else:
        min_ans = [min(channel) for channel in data]
    data = np.array(data)       # cast to numpy array
    for ich in range(len(data)):
        range_value = float(max_ans[ich]-min_ans[ich])
        if range_value == 0:
            normal_data = data[ich] - data[ich]
        else:
            normal_data = (data[ich] - min_ans[ich]) / range_value
        for inor in range(len(normal_data)):
            if normal_data[inor] < 0:
                normal_data[inor] = 0.0
        ans.append(normal_data)
    return ans, min_ans, max_ans

#filename = os. getcwd() + '/../data/train_data_09_17_2020_11_13_54.csv'
filename = os. getcwd() + '/../data/train_data_12_04_2020_14_09_01.csv'
data = []
with open(filename, newline='') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        try:
            data.append([float(value) for value in row])
        except:
            #print(row)
            pass

EMGdata = [item[:-6] for item in data]
Motiondata = [item[-6:] for item in data]

data_X = []
data_Y = []
tem_X = []
tem_Y = []
pre_j = Motiondata[0]

for i,j in zip(EMGdata,Motiondata):
    #print(j)
    if j != pre_j:
        data_X.append(tp(tem_X))
        data_Y.append(tp(tem_Y))
        tem_X = []
        tem_Y = []
    tem_X.append(i)
    tem_Y.append(j)
    pre_j = j

data_X.append(tp(tem_X))
data_Y.append(tp(tem_Y))

nEMGdata, minEMGdata, maxEMGdata = array_normalize(EMGdata)

# for i in range(len(data_X)):
#     data_X[i] = np.array(data_X[i])
# for i in range(len(data_Y)):
#     data_Y[i] = np.array(data_Y[i])

gripdata_X = np.concatenate((data_X[0], data_X[1], data_X[2]), axis=1)
gripdata_Y = np.concatenate((data_Y[0], data_Y[1], data_Y[2]), axis=1)

wristdata_X = np.concatenate((data_X[0], data_X[3], data_X[4]), axis=1)
wristdata_Y = np.concatenate((data_Y[0], data_Y[3], data_Y[4]), axis=1)

prosudata_X = np.concatenate((data_X[0], data_X[5], data_X[6]), axis=1)
prosudata_Y = np.concatenate((data_Y[0], data_Y[5], data_Y[6]), axis=1)

sum_x = [tp(gripdata_X), tp(wristdata_X), tp(prosudata_X)]
sum_y = [tp(gripdata_Y), tp(wristdata_Y), tp(prosudata_Y)]

syn = synergy()
syn.fit(sum_x, sum_y)
transform_y = syn.transform(EMGdata)
multiplot(transform_y)


x = torch.FloatTensor(transform_y)  # x data (tensor), shape=(100, 1)
y = torch.FloatTensor(Motiondata)                # noisy y data (tensor), shape=(100, 1) 

x, y = Variable(x), Variable(y)




tensor_y = torch.FloatTensor(transform_y)
tensor_motion = torch.FloatTensor(Motiondata)
cnn = cnnRegession(input_n = tensor_y.shape[1], output_n = tensor_motion.shape[1])
cnn.train(tensor_y, tensor_motion, BATCH_SIZE = 64, EPOCH = 5)

angle = cnn.predict(tensor_y)
multiplot(angle.detach().numpy()) 

my_range = range(len(angle))



filename = os. getcwd() + '/../data/run_data_12_04_2020_14_09_01.csv'
data = []
with open(filename, newline='') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        try:
            data.append([float(value) for value in row])
        except:
            #print(row)
            pass

EMGdata = [item[:-6] for item in data]
Motiondata = [item[-6:] for item in data]

data_X = []
data_Y = []
tem_X = []
tem_Y = []
pre_j = Motiondata[0]

for i,j in zip(EMGdata,Motiondata):
    #print(j)
    if j != pre_j:
        data_X.append(tp(tem_X))
        data_Y.append(tp(tem_Y))
        tem_X = []
        tem_Y = []
    tem_X.append(i)
    tem_Y.append(j)
    pre_j = j

data_X.append(tp(tem_X))
data_Y.append(tp(tem_Y))

transform_y = syn.transform(EMGdata)

tensor_y = torch.FloatTensor(transform_y)
angle = cnn.predict(tensor_y)
multiplot(angle.detach().numpy()) 
