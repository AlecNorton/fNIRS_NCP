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

#Since we want to tune for generalization, we split the dataset into subjects
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



train_subjects = 0
test_subjects = 0
x_train = pd.DataFrame()
x_test = pd.DataFrame()

#Load all data from each file except zero subjects
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



#Load the zero subjects one by one to test generalization. 
df = pd.read_csv('/home/arnorton/NCP_Testing/size_30sec_150ts_stride_03ts/sub_07.csv')
#Test data is first concated from the training data to include seen data. 
x_test = pd.concat([x_train, df])

#Further data is added to the test data. 
df = pd.read_csv('/home/arnorton/NCP_Testing/size_30sec_150ts_stride_03ts/sub_01.csv')
x_test = pd.concat([x_test, df])
#Last bit of data is being added to training data, but this could be added to test data, depends on current iteration of testing.
df = pd.read_csv('/home/arnorton/NCP_Testing/size_30sec_150ts_stride_03ts/sub_05.csv')
x_train = pd.concat([x_train, df])

#Change number of train_subjects vs. test_subjects based on above configuration. 
train_subjects = train_subjects + 1
test_subjects = test_subjects + 2



#Load the labeled data.
y_train = x_train.loc[:, ['chunk', 'label']]
x_train.pop('chunk')
x_train.pop('label')


x_train = np.array(x_train)
print(x_train.shape)
#Reshape based on the amount of samples in the window
reshape = int(x_train.shape[0]/150)
print(reshape)
x_train = x_train.reshape(reshape, 150, 8)

#You can choose to pre-process the data, in this case we abstain from doing so.
#x_train = (x_train - np.mean(x_train, axis = 0)) / np.std(x_train, axis = 0)

x_train = x_train.astype(np.float32)

y_train = np.array(y_train)
y_train = y_train.reshape(reshape, 150, 2)
array = np.zeros(reshape, )
for i in range(0, reshape - 1):
    array[i] = y_train[i][0][1]

y_train = array
y_train = y_train.astype(np.int8)


#Do the same as above but with the test data.
y_test = x_test.loc[:, ['chunk', 'label']]
x_test.pop('chunk')
x_test.pop('label')


x_test = np.array(x_test)
print(x_test.shape)
reshape = int(x_test.shape[0]/150)
print(reshape)
x_test = x_test.reshape(reshape, 150, 8)

#You can choose to pre-process the data, in this case we abstain from doing so. 
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

#Load a LTC NCP Model for hyperparemeter tuning
def LTC_NCP_model_builder(hp):
    '''
    inter_neuron = hp.Int('inter_neurons', min_value = 5, max_value = 30, step = 1)
    command_neuron = hp.Int('command_neurons', min_value = 5, max_value = 30, step = 1)
    motor_neuron = hp.Int('motor_neurons', min_value = 4, max_value = 30, step = 1)
    sensory_fanout = hp.Int('sensory_fanout', min_value = 1, max_value = int(.9 * inter_neuron), step = 1)
    inter_fanout = hp.Int('inter_fanout', min_value = 1, max_value = int(.9 * command_neuron), step = 1)
    recurrent_command_synapses = hp.Int('recurrent_command_synapses', min_value = 1, max_value = int(1.8 * command_neuron))
    motor_fanin = hp.Int('motor_fanin', min_value = 1, max_value = int(.9 * command_neuron), step = 1)
    
    wiring = ncps.wirings.NCP(inter_neurons = inter_neuron, command_neurons = command_neuron, motor_neurons = motor_neuron, sensory_fanout = sensory_fanout, inter_fanout = inter_fanout, recurrent_command_synapses= recurrent_command_synapses, motor_fanin= motor_fanin)
    '''
    #units = hp.Int('units', min_value = 10, max_value = 200, step = 10)
    #output_size = hp.Int('output_size', min_value = 5, max_value = units - 3, step = 10)
    #sparsity_level = hp.Float('sparsity_level', min_value = .1, max_value = .9, step = .1)
    
    #wiring = ncps.wirings.AutoNCP(units = units, output_size = output_size, sparsity_level = sparsity_level)
    wiring = ncps.wirings.AutoNCP(units = 100, output_size = 5, sparsity_level= .5)
    #backbone_units = hp.Int('backbone_units', min_value = 64, max_value = 256, step = 32)
    #backbone_layers = hp.Int('backbone_layer', min_value = 0, max_value = 3, step = 1)
    #backbone_dropout = hp.Float('backbone_dropout', min_value = 0, max_value = .9, step = .1)

    x = tf.keras.layers.Conv1D(32, 3)(input)
    x = tf.keras.layers.MaxPool1D(3)(x)   
    x = LTC(wiring, return_sequences= True)(x)
    x = tf.keras.layers.Flatten()(x)
    output = tf.keras.layers.Dense(4)(x)

    model = tf.keras.Model(inputs = input, outputs = output)

    hp_learning_rate = hp.Choice('learning_rate', values = [.001, .005, .01, .015, .02, .025, .03])
    #hp_learning_rate = .02
    #decay_lr = .66
    #hp_clipnorm = .9999
    hp_clipnorm = hp.Float('clipnorm', min_value = .1, max_value = 1, step = .1)
    train_steps = reshape // 64
    decay_lr = hp.Float('decay_lr', min_value = 0, max_value = 1, step = .1)



    learning_rate_fn = tf.keras.optimizers.schedules.ExponentialDecay(
        hp_learning_rate, train_steps, decay_lr
    )

    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate_fn, clipnorm = hp_clipnorm),
                  loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
                  metrics = ['accuracy'])
    
    return model

#Use hyperband, as this is a quick and effective method. Other possibilities for hypertuning are grid search where it goes over every possible parameter
#and bayesian optimization which takes a bit longer than hyperband but with generally better results.
tuner = kt.Hyperband(LTC_NCP_model_builder,
                     objective = 'val_accuracy',
                     max_epochs = 10,
                     factor = 3,
                     overwrite = True,
                     distribution_strategy=tf.distribute.MirroredStrategy(),
                     directory = '',
                     project_name = "LTC_NCP_Tuning_Project")

#If the model isn't improving over 5 batches, stop the testing and move onto a different iteration of the model
stop_early = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 5)

#Search for the best parameters for the model!
tuner.search(x_train, y_train, epochs = 50, validation_data = (x_test, y_test), callbacks = [stop_early])

best_hps = tuner.get_best_hyperparameters(num_trials = 1)[0]


#Rebuild the CNN model with these hyperparameters and re-train to determine the best epoch.
model = tuner.hypermodel.build(best_hps)
history = model.fit(x_train, y_train, epochs=20, validation_data = (x_test, y_test))

val_acc_per_epoch = history.history['val_accuracy']
best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
print('Best epoch: %d' % (best_epoch,))


hypermodel = tuner.hypermodel.build(best_hps)

hypermodel.summary()



# Retrain the model once again upto the best epoch and record results below. 
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

