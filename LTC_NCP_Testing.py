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


class CustomCallback(tf.keras.callbacks.Callback):
    def on_batch_end(self, batch, logs = None):
        if(logs["loss"] > 5000):
            self.model.stop_training = True
#Actual Execution of Code: 

#Load Data Here

#TODO: Load a Time-Series Application

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

batch_size = 64


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
    train_steps = reshape // batch_size
    decay_lr = hp.Float('decay_lr', min_value = 0, max_value = 1, step = .1)



    learning_rate_fn = tf.keras.optimizers.schedules.ExponentialDecay(
        hp_learning_rate, train_steps, decay_lr
    )

    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate_fn, clipnorm = hp_clipnorm),
                  loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
                  metrics = ['accuracy'])
    
    return model

tuner = kt.Hyperband(LTC_NCP_model_builder,
                     objective = 'val_accuracy',
                     max_epochs = 10,
                     overwrite = True,
                     directory = '',
                     distribution_strategy = tf.distribute.MirroredStrategy(),
                     project_name = "LTC_NCP_Tuning_Project")

stop_early = CustomCallback()
stop_early1 = tf.keras.callbacks.TerminateOnNaN()
stop_early2 = tf.keras.callbacks.EarlyStopping(monitor = 'loss', mode = "min", patience = 2)

print("Begin searching")

tuner.search(x_train, y_train, epochs = 5, validation_data = (x_test, y_test), callbacks = [stop_early, stop_early1, stop_early2], verbose = 1, batch_size = batch_size)

best_hps = tuner.get_best_hyperparameters(num_trials = 1)[0]


model = tuner.hypermodel.build(best_hps)
history = model.fit(x_train, y_train, epochs=20, validation_data = (x_test, y_test), verbose = 1, batch_size = batch_size)

val_acc_per_epoch = history.history['val_accuracy']
best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1


hypermodel = tuner.hypermodel.build(best_hps)




# Retrain the model
hypermodel.fit(x_train, y_train, epochs=best_epoch, validation_data = (x_test, y_test), verbose = 1, batch_size = batch_size)

eval_result = hypermodel.evaluate(x_test, y_test)

hypermodel.summary()


'''
print("LTC_NCP_Testing")
print(f"""
The hyperparameter search is complete. Optimal values below: 
      units = {best_hps.get('units')},
      output_size = {best_hps.get('output_size')},
      sparsity_level = {best_hps.get('sparsity_level')}




""")'''

print("LTC_NCP_Testing")
print(f"""
The hyperparameter search is complete. Optimal values below: 
      learning_rate = {best_hps.get('learning_rate')},
      clipnorm = {best_hps.get('clipnorm')},
      decay_lr = {best_hps.get('decay_lr')}




""")

print('Best epoch: %d' % (best_epoch,))
print("[test loss, test accuracy]:", eval_result)
print("150ts")