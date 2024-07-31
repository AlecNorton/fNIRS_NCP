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

#For Wiring
parser.add_argument("size")
parser.add_argument("output_size")
parser.add_argument("sparsity")

#For optimizer
parser.add_argument("base_lr")
parser.add_argument("clipnorm")

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

#define a function to return a NCP LTC Model
def LTC_NCP(input, ncp_size, ncp_output_size, ncp_sparsity_level):
    #Set up architecture for Neural Circuit Policy
    wiring = ncps.wirings.AutoNCP(ncp_size, ncp_output_size, ncp_sparsity_level)
    #Begin constructing layer, starting with input
    x = tf.keras.layers.Conv1D(32, 3)(input)
    x = tf.keras.layers.MaxPool1D(3)(x)
    x = LTC(wiring, return_sequences= True)(x)
    x = keras.layers.Flatten()(x)
    output = tf.keras.layers.Dense(4)(x)

    model = tf.keras.Model(inputs = input, outputs = output)
    
    
    #Return model
    return model

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

#CfC NCP
ncp_size = int(args.size)
ncp_output_size = int(args.output_size)
ncp_sparsity_level = float(args.sparsity)

#number_of_models = int(args.model_number)
batch_size = int(args.batch_size)
epochs = int(args.epochs)

base_lr = float(args.base_lr)
train_steps = reshape // batch_size
decay_lr = .66
clipnorm = float(args.clipnorm)


#Create exponential learning_rate function.
learning_rate_fn = tf.keras.optimizers.schedules.ExponentialDecay(
        base_lr, train_steps, decay_lr
    )

#Create optimizer
cfc_optimizer = tf.keras.optimizers.Adam(learning_rate_fn, clipnorm = clipnorm)

#Create loss function
cfc_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)

#Create k-fold Cross Validation
kf = KFold(n_splits = int(args.kfold), shuffle =True)

#Create model that we are validating
model = LTC_NCP(input, ncp_size, ncp_output_size, ncp_sparsity_level)
model.compile(cfc_optimizer, cfc_loss, metrics = tf.keras.metrics.SparseCategoricalAccuracy())


scores = []

#Begin k-fold cross validation
for train, test in kf.split(x_train, y_train):

    print(f"    Train: index = {train}")
    print(f"    Train: index = {test}")

    print("X_train")
    print(x_train[train])
    print("Y_Train")
    print(y_train[train])
    
    model = LTC_NCP(input, ncp_size, ncp_output_size, ncp_sparsity_level)

    cfc_optimizer = tf.keras.optimizers.Adam(learning_rate_fn, clipnorm = clipnorm)
    cfc_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)


    model.compile(cfc_optimizer, cfc_loss, metrics = tf.keras.metrics.SparseCategoricalAccuracy())


    model.fit(x_train[train], y_train[train], batch_size = batch_size, epochs = epochs)
    scores.append(model.evaluate(x_train[test], y_train[test])[1])
    model.save_weights('LTC_NCP_Model/saved_model.weights.h5')
    del model
    gc.collect()


print(scores)
print("Average: ")
print(np.mean(scores))

#Print results of k-fold cross validation
print("LTC-NCP Cross Fold Training: 150ts")
print("\n")
print("base_lr = " + str(base_lr) + " decay_lr = " + str(decay_lr) + " clipnorm = " + str(clipnorm))
print("\n")
print("Size of Model: " + str(ncp_size) + " Output Size Of Model: " + str(ncp_output_size) + " NCP Sparsity Level: " + str(ncp_sparsity_level))
print("\n")
print("Epochs: " + str(epochs) + " Batch Size: " + str(batch_size) + " Number Of Folds: " + str(args.kfold))
