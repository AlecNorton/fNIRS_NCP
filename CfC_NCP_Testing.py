import sys
import argparse




parser = argparse.ArgumentParser()

#For Wiring
parser.add_argument("size")
parser.add_argument("output_size")
parser.add_argument("sparsity")

#For opt
parser.add_argument("base_lr")
parser.add_argument("clipnorm")
#CfC args
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


class CustomCallback(tf.keras.callbacks.Callback):
    def on_batch_end(self, batch, logs = None):
        if(logs["loss"] > 5000):
            self.model.stop_training = True

keras = tf.keras
#define a function to return a NCP CfC Model
def LTC_NCP(input, ncp_size, ncp_output_size, ncp_sparsity_level):
    #Set up architecture for Neural Circuit Policy
    wiring = ncps.wirings.AutoNCP(ncp_size, ncp_output_size, ncp_sparsity_level)
    #Begin constructing layer, starting with input
    
    '''model = tf.keras.models.Sequential(
        [
            keras.layers.InputLayer(input_shape = (None, 8)),
            CfC(wiring),
            tf.keras.layers.Dense(1)
        ]
    )'''
    x = tf.keras.layers.Conv1D(32, 3)(input)
    x = tf.keras.layers.MaxPool1D(3)(x)
    x = LTC(wiring, return_sequences= True)(x)
    x = keras.layers.Flatten()(x)
    output = tf.keras.layers.Dense(4)(x)

    model = tf.keras.Model(inputs = input, outputs = output)
    
    
    #Return model
    return model

def LTC_FC(input, ncp_size, ncp_output_size):
    #Set up architecture for Neural Circuit Policy
    wiring = ncps.wirings.FullyConnected(ncp_size, ncp_output_size)
    #Begin constructing layer, starting with input
    
    '''model = tf.keras.models.Sequential(
        [
            keras.layers.InputLayer(input_shape = (None, 8)),
            CfC(wiring),
            tf.keras.layers.Dense(1)
        ]
    )'''
    x = tf.keras.layers.Conv1D(32, 3)(input)
    x = tf.keras.layers.MaxPool1D(3)(x)
    x = LTC(wiring, return_sequences= True)(x)
    x = keras.layers.Flatten()(x)
    output = tf.keras.layers.Dense(4)(x)

    model = tf.keras.Model(inputs = input, outputs = output)
    
    
    #Return model
    return model

def CNN(input):
    #Algorithm make up is CNN2-a from Trakoolqilaiwan et all.
    #x = tf.keras.layers.Conv2D(32, 3)(input)
    x = tf.keras.layers.Conv1D(64, 3)(input)
    x = tf.keras.layers.MaxPool1D(2)(x)
    x = tf.keras.layers.Dropout(.3)(x)

    x = tf.keras.layers.Conv1D(64, 3)(input)
    x = tf.keras.layers.MaxPool1D(2)(x)
    x = tf.keras.layers.Dropout(.3)(x)

    x = tf.keras.layers.Conv1D(64, 3)(input)
    x = tf.keras.layers.MaxPool1D(2)(x)
    x = tf.keras.layers.Dropout(.3)(x)

    x = tf.keras.layers.Flatten()(x)

    x = tf.keras.layers.Dense(480, activation = "relu")(x)
    x = tf.keras.layers.Dense(480, activation = "relu")(x)

    output = tf.keras.layers.Dense(4)(x)
    
    return tf.keras.Model(inputs = input, outputs = output)


def eval(model, index_arg, train_x, train_y, opt, loss_fun, batch_size, epochs):
    #Compile the Model
    model.compile(optimizer = opt, loss = loss_fun, metrics = tf.keras.metrics.SparseCategoricalAccuracy())

    #Return a summary of the model and its (non)trainable paramaters
    model.summary()

    #Fit the model and return accuracy
    #Get beginning of time
    callback = tf.keras.callbacks.EarlyStopping(monitor = 'loss', patience = 3)
    #callback1 = CustomCallback()
    callback2 = tf.keras.callbacks.TerminateOnNaN()


    start = time.process_time()
    hist = model.fit(x = train_x, y = train_y, validation_split = .33, batch_size = batch_size, epochs = epochs, verbose = 1, callbacks = [callback, callback2])
    end = time.process_time()
    test_accuracies = hist.history["val_sparse_categorical_accuracy"]
    print("Max Accuracy Of Model: " + str(np.max(test_accuracies)))
    return np.max(test_accuracies), end-start, model

#Based on the model_number, create a model and train on specified optimizer, loss_function, validation_split, batch_size, and some epochs
#Then return the mean and standard deviation of the accuracy of these models. 
def score(model, train_x, train_y, x_test, y_test, opt, loss_fun, model_number, batch_size, epochs):
    acc = []
    dur = []
    for i in range(model_number):
        print("Model: " + str(i))
        max_accuracy, time, model = eval(model, i, train_x, train_y, opt, loss_fun, batch_size, epochs)
        dur.append(time)
        acc.append(100 * max_accuracy)
    acc_average = np.mean(acc)
    acc_std = np.std(acc)
    dur_average = np.mean(dur)
    dur_std = np.std(dur)
    print("-------------------------------------------------------------------")
    print("Average Test Accuracy: " + str(acc_average) + " Standard Deviation Test Accuracy: " + str(acc_std))
    print("Average Time Training: " + str(dur_average) + " Standard Deviation Time: " + str(dur_std))
    print("-------------------------------------------------------------------")

    results = model.evaluate(x_test, y_test, batch_size = 32)
    print("test loss, test acc:", results)

    model.summary()
    


    #print(f"Test Accuracy: {np.mean(acc):(1/model_number)}\\% $\\pm$ {np.std(acc):(1/model_number)}")


#Actual Execution of Code: 

#Load Data Here

#TODO: Load a Time-Series Application

csv_files = glob.glob('/home/arnorton/NCP_Testing/size_02sec_10ts_stride_03ts/*.csv')
#csv_files2 = glob.glob('size_30sec_150ts_stride_03ts/sub_3*.csv')

x_train = pd.DataFrame()
for csv_file in csv_files:
    df = pd.read_csv(csv_file)
    x_train = pd.concat([x_train, df])



'''
csv_file1 = pd.read_csv('size_30sec_150ts_stride_03ts\sub_97.csv')
csv_file2 = pd.read_csv('size_30sec_150ts_stride_03ts\sub_1.csv')
csv_file3 = pd.read_csv('size_30sec_150ts_stride_03ts\sub_15.csv')
csv_file4 = pd.read_csv('size_30sec_150ts_stride_03ts\sub_31.csv')
csv_file5 = pd.read_csv('size_30sec_150ts_stride_03ts\sub_45.csv')

x_train = csv_file1.copy()
x_train = pd.concat([x_train, csv_file2])
x_train = pd.concat([x_train, csv_file3])
x_train = pd.concat([x_train, csv_file4])
x_train = pd.concat([x_train, csv_file5])
'''

y_train = x_train.loc[:, ['chunk', 'label']]
x_train.pop('chunk')
x_train.pop('label')


x_train = np.array(x_train)
print(x_train.shape)
reshape = int(x_train.shape[0]/10)
print(reshape)
x_train = x_train.reshape(reshape, 10, 8)

x_train = (x_train - np.mean(x_train, axis = 0)) / np.std(x_train, axis = 0)

x_train = x_train.astype(np.float32)

y_train = np.array(y_train)
y_train = y_train.reshape(reshape, 10, 2)
array = np.zeros(reshape, )
for i in range(0, reshape - 1):
    array[i] = y_train[i][0][1]

y_train = array
y_train = y_train.astype(np.int8)

#x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size = .33, shuffle = True)

input = tf.keras.layers.Input(shape = (10, 8))

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



learning_rate_fn = tf.keras.optimizers.schedules.ExponentialDecay(
        base_lr, train_steps, decay_lr
    )


cfc_optimizer = tf.keras.optimizers.Adam(learning_rate_fn, clipnorm = clipnorm)

cfc_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)

cnn_optimizer = tf.keras.optimizers.Adam()
cnn_loss_fun = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)



kf = KFold(n_splits = int(args.kfold), shuffle =True)

model = LTC_NCP(input, ncp_size, ncp_output_size, ncp_sparsity_level)
model.compile(cfc_optimizer, cfc_loss, metrics = tf.keras.metrics.SparseCategoricalAccuracy())


scores = []


for train, test in kf.split(x_train, y_train):

    print(f"    Train: index = {train}")
    print(f"    Train: index = {test}")

    print("X_train")
    print(x_train[train])
    print("Y_Train")
    print(y_train[train])
    
    model = LTC_NCP(input, ncp_size, ncp_output_size, ncp_sparsity_level)
    #model = LTC_FC(input, 100, 5)
    #model = CNN(input)

    cfc_optimizer = tf.keras.optimizers.Adam(learning_rate_fn, clipnorm = clipnorm)
    cfc_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)

    #cnn_optimizer = tf.keras.optimizers.Adam()
    #cnn_loss_fun = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)

    model.compile(cfc_optimizer, cfc_loss, metrics = tf.keras.metrics.SparseCategoricalAccuracy())
    #model.compile(cnn_optimizer, cnn_loss_fun, metrics = tf.keras.metrics.SparseCategoricalAccuracy())


    model.fit(x_train[train], y_train[train], batch_size = batch_size, epochs = epochs)
    scores.append(model.evaluate(x_train[test], y_train[test])[1])
    del model
    gc.collect()


print(scores)
print("Average: ")
print(np.mean(scores))




#x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size = .33, shuffle = True)

#score(LTC_NCP(input, ncp_size, ncp_output_size, ncp_sparsity_level), x_train, y_train, x_valid, y_valid, cfc_optimizer, cfc_loss, 1, batch_size, epochs)












#score(CFC_NCP(input, ncp_size, ncp_output_size, ncp_sparsity_level), x_train, y_train, x_test, y_test, cfc_optimizer, cfc_loss, number_of_models, batch_size, epochs)

print("LTC-NCP Cross Fold Training: 10ts")
print("\n")
print("base_lr = " + str(base_lr) + " decay_lr = " + str(decay_lr) + " clipnorm = " + str(clipnorm))
print("\n")
print("Size of Model: " + str(ncp_size) + " Output Size Of Model: " + str(ncp_output_size) + " NCP Sparsity Level: " + str(ncp_sparsity_level))
print("\n")
print("Epochs: " + str(epochs) + " Batch Size: " + str(batch_size) + " Number Of Folds: " + str(args.kfold))
