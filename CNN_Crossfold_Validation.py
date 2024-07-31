'''huangfNIRS2MW2021,
    title = {The Tufts fNIRS Mental Workload Dataset & Benchmark for Brain-Computer Interfaces that Generalize},
    booktitle = {Proceedings of the Neural Information Processing Systems (NeurIPS) Track on Datasets and Benchmarks},
    author = {Huang, Zhe and Wang, Liang and Blaney, Giles and Slaughter, Christopher and McKeon, Devon and Zhou, Ziyu and Jacob, Robert J. K. and Hughes, Michael C.},
    year = {2021},
    url = {https://openreview.net/pdf?id=QzNHE7QHhut},
}
'''

'''hasani_closed-form_2022,
	title = {Closed-form continuous-time neural networks},
	journal = {Nature Machine Intelligence},
	author = {Hasani, Ramin and Lechner, Mathias and Amini, Alexander and Liebenwein, Lucas and Ray, Aaron and Tschaikowski, Max and Teschl, Gerald and Rus, Daniela},
	issn = {2522-5839},
	month = nov,
	year = {2022},
}
'''


import sys
import argparse



#Get arguments from terminal
parser = argparse.ArgumentParser()



#Training and cross-fold arguments args
parser.add_argument("batch_size")
parser.add_argument("epochs")
parser.add_argument("kfold")

args = parser.parse_args()

import tensorflow as tf
import pandas as pd
import numpy as np
import ncps 
from ncps.tf import CfC
from ncps.tf import LTC
import matplotlib.pyplot as plt
import glob
import time 
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import train_test_split
import gc

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

keras = tf.keras

#define a function to return a CNN Model
def CNN(input):
    #Algorithm make up is CNN2-a from Trakoolqilaiwan et all.
    #x = tf.keras.layers.Conv2D(32, 3)(input)
    x = tf.keras.layers.Conv1D(32, 3)(input)
    x = tf.keras.layers.MaxPool1D(3)(x)
    x = tf.keras.layers.Dropout(.5)(x)

    x = tf.keras.layers.Conv1D(32, 3)(x)
    x = tf.keras.layers.MaxPool1D(3)(x)
    x = tf.keras.layers.Dropout(.5)(x)

    x = tf.keras.layers.Conv1D(32, 3)(x)
    x = tf.keras.layers.MaxPool1D(3)(x)
    x = tf.keras.layers.Dropout(.5)(x)

    x = tf.keras.layers.Flatten()(x)

    x = tf.keras.layers.Dense(384, activation = "relu")(x)
    x = tf.keras.layers.Dense(384, activation = "relu")(x)

    output = tf.keras.layers.Dense(4)(x)
    
    return tf.keras.Model(inputs = input, outputs = output)
#Actual Execution of Code: 

#Load Data Here
csv_files = glob.glob('/home/arnorton/NCP_Testing/size_30sec_150ts_stride_03ts/*.csv')

x_train = pd.DataFrame()
for csv_file in csv_files:
    df = pd.read_csv(csv_file)
    x_train = pd.concat([x_train, df])



#Grab labeled data from dataset
y_train = x_train.loc[:, ['chunk', 'label']]
x_train.pop('chunk')
x_train.pop('label')

#Make into a numpy array.
x_train = np.array(x_train)
print(x_train.shape)
#Reshape, 150 is based on the amount of samples in one window, in this case 150. 
reshape = int(x_train.shape[0]/150)
print(reshape)
x_train = x_train.reshape(reshape, 150, 8)
#Pre-process so each value is between the values of 0 and 1. Makes training the model faster. 
x_train = (x_train - np.mean(x_train, axis = 0)) / np.std(x_train, axis = 0)
#Ensures all variables are the same type. 
x_train = x_train.astype(np.float32)


#Do the same with labeled data as did with samples. 
y_train = np.array(y_train)
y_train = y_train.reshape(reshape, 150, 2)
array = np.zeros(reshape, )
for i in range(0, reshape - 1):
    array[i] = y_train[i][0][1]

y_train = array
y_train = y_train.astype(np.int8)


input = tf.keras.layers.Input(shape = (150, 8))


batch_size = int(args.batch_size)
epochs = int(args.epochs)




#Create optimizer
cnn_optimizer = tf.keras.optimizers.Adam()

#Create loss function
cnn_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)

#Create k-fold Cross Validation
kf = KFold(n_splits = int(args.kfold), shuffle =True)

#Create model that we are validating
model = CNN(input)
model.compile(cnn_optimizer, cnn_loss, metrics = tf.keras.metrics.SparseCategoricalAccuracy())


scores = []

#Begin k-fold cross validation
for train, test in kf.split(x_train, y_train):

    print(f"    Train: index = {train}")
    print(f"    Train: index = {test}")

    print("X_train")
    print(x_train[train])
    print("Y_Train")
    print(y_train[train])
    
    model = CNN(input)

    cnn_optimizer = tf.keras.optimizers.Adam()
    cnn_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)


    model.compile(cnn_optimizer, cnn_loss, metrics = tf.keras.metrics.SparseCategoricalAccuracy())


    model.fit(x_train[train], y_train[train], batch_size = batch_size, epochs = epochs)
    scores.append(model.evaluate(x_train[test], y_train[test])[1])
    #Make sure to have a directory called CNN_Model
    model.save('CNN_Model/saved_model')
    del model
    gc.collect()


print(scores)
print("Average: ")
print(np.mean(scores))

#Print results of k-fold cross validation
print("CNN Cross Fold Training: 150ts")
print("\n")
print("Epochs: " + str(epochs) + " Batch Size: " + str(batch_size) + " Number Of Folds: " + str(args.kfold))


