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

csv_files = glob.glob('/home/arnorton/NCP_Testing/size_02sec_10ts_stride_03ts/*.csv')


x_train = pd.DataFrame()
for csv_file in csv_files:
    df = pd.read_csv(csv_file)
    x_train = pd.concat([x_train, df])




#csv_file = pd.read_csv('size_30sec_150ts_stride_03ts\sub_1.csv')
#x_train = csv_file.copy()

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

x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size = .33, shuffle = True)

batch_size = 512

input = tf.keras.layers.Input(shape = (10, 8))

def CfC_NCP_model_builder(hp):
    '''
    inter_neuron = hp.Int('inter_neurons', min_value = 5, max_value = 30, step = 1)
    command_neuron = hp.Int('command_neurons', min_value = 5, max_value = 30, step = 1)
    motor_neuron = hp.Int('motor_neurons', min_value = 4, max_value = 30, step = 1)
    sensory_fanout = hp.Int('sensory_fanout', min_value = 1, max_value = int(.9 * inter_neuron), step = 1)
    inter_fanout = hp.Int('inter_fanout', min_value = 1, max_value = int(.9 * command_neuron), step = 1)
    recurrent_command_synapses = hp.Int('recurrent_command_synapses', min_value = 1, max_value = int(1.8 * command_neuron))
    motor_fanin = hp.Int('motor_fanin', min_value = 1, max_value = int(.9 * command_neuron), step = 1)
    '''
    
    #units = hp.Int('units', min_value = 50, max_value = 200, step = 2)
    #output_size = hp.Int('output_size', min_value = 5, max_value = units - 3, step = 2)
    #sparsity_level = hp.Float('sparsity_level', min_value = .1, max_value = .9, step = .1)
    #wiring = ncps.wirings.AutoNCP(units = units, output_size = output_size, sparsity_level = sparsity_level)

    #mode = hp.Choice('mode', values = ["default", "pure", "no_gate"])
    #ackbone_activation = hp.Choice('backbone_activation', values = ["silu", "relu", "tanh", "lecun_tanh", "softplus"])
    
    #batch_size = hp.Int('batch_size', min_value = 32, max_value = 256, step = 32)

    #backbone_units = hp.Int('backbone_units', min_value = 64, max_value = 256, step = 32)
    #backbone_layers = hp.Int('backbone_layer', min_value = 0, max_value = 3, step = 1)
    #backbone_dropout = hp.Float('backbone_dropout', min_value = 0, max_value = .9, step = .1)
    wiring = ncps.wirings.AutoNCP(units = 194, output_size = 21, sparsity_level = .5)
    
    x = tf.keras.layers.Conv1D(32, 3)(input)
    x = tf.keras.layers.MaxPool1D(3)(x)
    x = LTC(wiring, return_sequences= True)(x)
    x = tf.keras.layers.Flatten()(x)
    output = tf.keras.layers.Dense(4)(x)

    model = tf.keras.Model(inputs = input, outputs = output)

    hp_learning_rate = hp.Choice('learning_rate', values = [.001, .005, .01, .015, .02])
    clipnorm = hp.Float('clipnorm', min_value = .1, max_value = 1, step = .3)

    train_steps = reshape // batch_size
    decay_lr = hp.Float('decay_lr', min_value = 0, max_value = 1, step = .25)



    learning_rate_fn = tf.keras.optimizers.schedules.ExponentialDecay(
        hp_learning_rate, train_steps, decay_lr
    )
    #clipnorm = .9999

    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate_fn, clipnorm = clipnorm),
                  loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
                  metrics = ['accuracy'])
    
    return model

'''tuner = kt.Hyperband(CfC_NCP_model_builder,
                     objective = 'val_accuracy',
                     max_epochs = 5,
                     factor = 3,
                     overwrite = True,
                     distribution_strategy=tf.distribute.MirroredStrategy(),
                     directory = '',
                     project_name = "CfC_NCP_Tuning_Project")
 '''
tuner = kt.GridSearch(CfC_NCP_model_builder, objective = 'val_accuracy', max_trials = 20)                   

stop_early = CustomCallback()
stop_early1 = tf.keras.callbacks.TerminateOnNaN()
stop_early2 = tf.keras.callbacks.EarlyStopping('loss', mode = "min", patience = 5)

tuner.search(x_train, y_train, epochs = 5, validation_data = (x_valid, y_valid), callbacks = [stop_early, stop_early1, stop_early2], verbose = 1, batch_size = batch_size)

best_hps = tuner.get_best_hyperparameters(num_trials = 1)[0]


model = tuner.hypermodel.build(best_hps)
history = model.fit(x_train, y_train, epochs=20, validation_data = (x_valid, y_valid), verbose = 1, batch_size = batch_size)

val_acc_per_epoch = history.history['val_accuracy']
best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1


hypermodel = tuner.hypermodel.build(best_hps)




# Retrain the model
hypermodel.fit(x_train, y_train, epochs=best_epoch, validation_data = (x_valid, y_valid), verbose = 1, batch_size = batch_size)
eval_result = hypermodel.evaluate(x_valid, y_valid)

hypermodel.summary()

print("LTC_NCP ")
'''
print(f"""
The hyperparameter search is complete. Optimal values below: 
      inter neurons = {best_hps.get('inter_neurons')},
      command neurons = {best_hps.get('command_neurons')},
      motor neurons = {best_hps.get('motor_neurons')},
      sensory fanout = {best_hps.get('sensory_fanout')},
      inter_fanout = {best_hps.get('inter_fanout')},
      recurrent_command_synapses = {best_hps.get('recurrent_command_synapses')},
      motor_fanin = {best_hps.get('motor_fanin')},
      mode = {best_hps.get('mode')},
      backbone_activation = {best_hps.get('backbone_activation')},
      learning_rate = {best_hps.get('learning_rate')},



""")
'''
print(f"""
The hyperparameter search is complete. Optimal values below: 
      learning_rate = {best_hps.get('learning_rate')},
      clipnorm = {best_hps.get('clipnorm')},
      decay_lr = {best_hps.get('decay_lr')}



""")
print('Best epoch: %d' % (best_epoch,))

print("[test loss, test accuracy]:", eval_result)
