import tensorflow as tf

import pandas as pd
import numpy as np
import ncps 
from ncps.tf import CfC
from ncps.tf import LTC
import matplotlib.pyplot as plt
import glob
import time 
from sklearn.model_selection import train_test_split
import keras_tuner as kt

#Actual Execution of Code: 

#Load Data Here

#TODO: Load a Time-Series Application

csv_files = glob.glob('/home/arnorton/NCP_Testing/size_02sec_10ts_stride_03ts/*.csv')


csv_files = glob.glob('/home/arnorton/NCP_Testing/size_30sec_150ts_stride_03ts/*.csv')
zero_subjects = glob.glob('/home/arnorton/NCP_Testing/size_30sec_150ts_stride_03ts/sub_0*.csv')
one_subjects = glob.glob('/home/arnorton/NCP_Testing/size_30sec_150ts_stride_03ts/sub_1*.csv')
two_subjects = glob.glob('/home/arnorton/NCP_Testing/size_30sec_150ts_stride_03ts/sub_2*.csv')
three_subjects = glob.glob('/home/arnorton/NCP_Testing/size_30sec_150ts_stride_03ts/sub_3*.csv')
four_subjects = glob.glob('/home/arnorton/NCP_Testing/size_30sec_150ts_stride_03ts/sub_4*.csv')
five_subjects = glob.glob('/home/arnorton/NCP_Testing/size_30sec_150ts_stride_03ts/sub_5*.csv')
six_subjects = glob.glob('/home/arnorton/NCP_Testing/size_30sec_150ts_stride_03ts/sub_6*.csv')
seven_subjects = glob.glob('/home/arnorton/NCP_Testing/size_30sec_150ts_stride_03ts/sub_7*.csv')
eight_subjects = glob.glob('/home/arnorton/NCP_Testing/size_30sec_150ts_stride_03ts/sub_8*.csv')
nine_subjects = glob.glob('/home/arnorton/NCP_Testing/size_30sec_150ts_stride_03ts/sub_9*.csv')


'''
x_train = pd.DataFrame()
for csv_file in csv_files:
    df = pd.read_csv(csv_file)
    x_train = pd.concat([x_train, df])

'''
train_subjects = 0
test_subjects = 0
x_train = pd.DataFrame()
x_test = pd.DataFrame()


for csv_file in one_subjects:
    df = pd.read_csv(csv_file)
    x_train = pd.concat([x_train, df])
    train_subjects = train_subjects + 1
for csv_file in two_subjects:
    df = pd.read_csv(csv_file)
    x_train = pd.concat([x_train, df])
    train_subjects = train_subjects + 1
for csv_file in three_subjects:
    df = pd.read_csv(csv_file)
    x_train = pd.concat([x_train, df])
    train_subjects = train_subjects + 1
for csv_file in four_subjects:
    df = pd.read_csv(csv_file)
    x_train = pd.concat([x_train, df])
    train_subjects = train_subjects + 1
for csv_file in five_subjects:
    df = pd.read_csv(csv_file)
    x_train = pd.concat([x_train, df])
    train_subjects = train_subjects + 1
for csv_file in six_subjects:
    df = pd.read_csv(csv_file)
    x_train = pd.concat([x_train, df])
    train_subjects = train_subjects + 1
for csv_file in seven_subjects:
    df = pd.read_csv(csv_file)
    x_train = pd.concat([x_train, df])
    train_subjects = train_subjects + 1
for csv_file in eight_subjects:
    df = pd.read_csv(csv_file)
    x_train = pd.concat([x_train, df])
    train_subjects = train_subjects + 1
for csv_file in nine_subjects:
    df = pd.read_csv(csv_file)
    x_train = pd.concat([x_train, df])
    train_subjects = train_subjects + 1



df = pd.read_csv('/home/arnorton/NCP_Testing/size_30sec_150ts_stride_03ts/sub_07.csv')
x_test = pd.concat([x_test, df])
#df = pd.read_csv('/home/arnorton/NCP_Testing/size_30sec_150ts_stride_03ts/sub_01.csv')
#x_test = pd.concat([x_test, df])
#df = pd.read_csv('/home/arnorton/NCP_Testing/size_30sec_150ts_stride_03ts/sub_05.csv')
#x_test = pd.concat([x_test, df])

df = pd.read_csv('/home/arnorton/NCP_Testing/size_30sec_150ts_stride_03ts/sub_01.csv')
x_train = pd.concat([x_train, df])
df = pd.read_csv('/home/arnorton/NCP_Testing/size_30sec_150ts_stride_03ts/sub_05.csv')
x_train = pd.concat([x_train, df])
train_subjects = train_subjects + 2
test_subjects = test_subjects + 1



y_train = x_train.loc[:, ['chunk', 'label']]
x_train.pop('chunk')
x_train.pop('label')


x_train = np.array(x_train)
print(x_train.shape)
reshape = int(x_train.shape[0]/150)
print(reshape)
x_train = x_train.reshape(reshape, 150, 8)

#x_train = (x_train - np.mean(x_train, axis = 0)) / np.std(x_train, axis = 0)

x_train = x_train.astype(np.float32)

y_train = np.array(y_train)
y_train = y_train.reshape(reshape, 150, 2)
array = np.zeros(reshape, )
for i in range(0, reshape - 1):
    array[i] = y_train[i][0][1]

y_train = array
y_train = y_train.astype(np.int8)



y_test = x_test.loc[:, ['chunk', 'label']]
x_test.pop('chunk')
x_test.pop('label')


x_test = np.array(x_test)
print(x_test.shape)
reshape = int(x_test.shape[0]/150)
print(reshape)
x_test = x_test.reshape(reshape, 150, 8)

#x_test = (x_test - np.mean(x_test, axis = 0)) / np.std(x_test, axis = 0)

x_test = x_test.astype(np.float32)

y_test = np.array(y_test)
y_test = y_test.reshape(reshape, 150, 2)
array = np.zeros(reshape, )
for i in range(0, reshape - 1):
    array[i] = y_test[i][0][1]

y_test = array
y_test = y_test.astype(np.int8)




input = tf.keras.layers.Input(shape = (150, 8))


def CNN_model_builder(hp):
    
    hp_c1 = hp.Int('conv units', min_value=32, max_value = 128, step = 32)
    hp_p1 = hp.Int('pool kernel', min_value = 2, max_value = 5, step = 1)
    hp_d1 = hp.Float('dropout rate', min_value = .1, max_value = .9, step = .1)
    x = tf.keras.layers.Conv1D(filters = hp_c1, kernel_size = 3)(input)
    x = tf.keras.layers.MaxPool1D(pool_size = hp_p1)(x)
    x = tf.keras.layers.Dropout(rate = hp_d1)(x)

    x = tf.keras.layers.Conv1D(filters = hp_c1, kernel_size = 3)(x)
    x = tf.keras.layers.MaxPool1D(pool_size = hp_p1)(x)
    x = tf.keras.layers.Dropout(rate = hp_d1)(x)

    x = tf.keras.layers.Conv1D(filters = hp_c1, kernel_size = 3)(x)
    x = tf.keras.layers.MaxPool1D(pool_size = 2)(x)
    x = tf.keras.layers.Dropout(rate = hp_d1)(x)

    x = tf.keras.layers.Flatten()(x)

    hp_dense1 = hp.Int('1st dense units', min_value = 32, max_value = 512, step = 64)
    hp_dense2 = hp.Int('2nd dense units', min_value = 32, max_value = 512, step = 64)

    x = tf.keras.layers.Dense(hp_dense1, activation = "relu")(x)
    x = tf.keras.layers.Dense(hp_dense2, activation = "relu")(x)

    output = tf.keras.layers.Dense(4)(x)

    model = tf.keras.Model(inputs = input, outputs = output)

    hp_learning_rate = hp.Choice('learning_rate', values = [.001, .005, .01])

    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = hp_learning_rate),
                  loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
                  metrics = ['accuracy'])
    
    return model

tuner = kt.Hyperband(CNN_model_builder,
                     objective = 'val_accuracy',
                     max_epochs = 10,
                     factor = 3,
                     overwrite = True,
                     distribution_strategy=tf.distribute.MirroredStrategy(),
                     directory = '',
                     project_name = "CNN_Tuning_Project")

stop_early = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 5)

tuner.search(x_train, y_train, epochs = 50, validation_data = (x_test, y_test), callbacks = [stop_early])

best_hps = tuner.get_best_hyperparameters(num_trials = 1)[0]



model = tuner.hypermodel.build(best_hps)
history = model.fit(x_train, y_train, epochs=20, validation_data = (x_test, y_test))

val_acc_per_epoch = history.history['val_accuracy']
best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
print('Best epoch: %d' % (best_epoch,))


hypermodel = tuner.hypermodel.build(best_hps)

hypermodel.summary()



# Retrain the model
hypermodel.fit(x_train, y_train, epochs=best_epoch, validation_data = (x_test, y_test))

eval_result = hypermodel.evaluate(x_test, y_test)
print("[test loss, test accuracy]:", eval_result)


print(f"""
The hyperparameter search is complete. The optimal number of units in the conv layer
layer is {best_hps.get('conv units')} and the best pool kernel is {best_hps.get('pool kernel')},
and the best dropout rate is {best_hps.get('dropout rate')}, and the best number of neurons in the dense layer is
{best_hps.get('1st dense units')}, and the best  number of neurons in the second dense layer is {best_hps.get('2nd dense units')}
and the best optimal learning rate for the optimizer
is {best_hps.get('learning_rate')}. 
""")

print("150ts")
print("Full amount of layers")

