#-----------------------------------------------------------------
#-- Master Thesis - Model-Based Predictive Maintenance on FPGA
#--
#-- File : training.py
#-- Description : Model design and training module
#--
#-- Author : Sami Foery
#-- Master : MSE Mechatronics
#-- Date : 14.01.2022
#-----------------------------------------------------------------

import data_structure as ds
import configparser
import datetime
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.callbacks import TensorBoard
from model_manipulation import Model
from matplotlib import pyplot as plt

print("All libraries are loaded from training")

# fit a customized LSTM network to training data
def fit_lstm(train, test,
             batch_size,
             nb_epoch,
             timestep,
             neurons_l1, neurons_l2, neurons_l3,
             tensorboard_callback):

    X = train
    Y = test
    X = X.reshape(int(X.shape[0]/timestep), timestep, X.shape[1])  # reshape into [samples, time steps, features]
    Y = Y.reshape(int(Y.shape[0]/timestep), timestep, Y.shape[1])
    model = Sequential()

    # hidden layers customization
    model.add(LSTM(neurons_l2, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
    model.add(LSTM(neurons_l3, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
    model.add(LSTM(neurons_l2, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
    model.add(Dense(1))

    model.compile(loss='mae', optimizer='adam')
    model.summary()
    hist = model.fit(X, X,
                     epochs=nb_epoch,
                     batch_size=batch_size,
                     verbose="auto",
                     validation_split=0.1,
                     shuffle=True,
                     callbacks=[tensorboard_callback])


    return model, hist

# fit a customized DENSE network to training data
def fit_dense(train, test,
              batch_size,
              nb_epoch,
              timestep,
              neurons_l1, neurons_l2,
              tensorboard_callback):

    X = train
    Y = test
    X = X.reshape(int(X.shape[0]/timestep), timestep, X.shape[1])  # reshape into [samples, time steps, features]
    Y = Y.reshape(int(Y.shape[0]/timestep), timestep, Y.shape[1])
    model = Sequential()
    model.add(Dense(neurons_l1, activation='sigmoid', input_shape=(X.shape[1], X.shape[2])))
    model.add(Dense(neurons_l2, activation='sigmoid', input_shape=(X.shape[1], X.shape[2])))
    model.add(Dense(neurons_l2, activation='sigmoid', input_shape=(X.shape[1], X.shape[2])))
    model.add(Dense(neurons_l1, activation='sigmoid', input_shape=(X.shape[1], X.shape[2])))
    model.add(Dense(1))

    model.compile(loss='mae', optimizer='adam')
    model.summary()
    hist = model.fit(X, X,
                     epochs=nb_epoch,
                     batch_size=batch_size,
                     verbose="auto",
                     validation_split=0.1,
                     shuffle=True,
                     callbacks=[tensorboard_callback])

    return model, hist

#  plot the cost function results from training
def cost_function(hist):
    plt.figure(figsize=(16, 9))
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('loss MAE')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


# load configurations
parser = configparser.ConfigParser()
parser.read('network_config.ini')

# variables assignation
batch_size = parser.getint('NETWORK', 'BATCH_SIZE')
nb_epochs = parser.getint('NETWORK', 'EPOCHS')
timestep = parser.getint('DATA', 'TIMESTEP')
neurons_l1 = parser.getint('NETWORK', 'NEURONS_L1')
neurons_l2 = parser.getint('NETWORK', 'NEURONS_L2')
neurons_l3 = parser.getint('NETWORK', 'NEURONS_L3')
model_dir = parser.get('MODEL', 'MODEL_DIR')
model_name = parser.get('MODEL', 'MODEL_NAME')
LSTM_MODEL = True

# define tensorboard callback
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

# call the fit function for LSTM or Dense model
if LSTM_MODEL:
    model, hist = fit_lstm(
        ds.train_scaled,
        ds.test_scaled,
        batch_size,
        nb_epochs,
        timestep,
        neurons_l1, neurons_l2, neurons_l3,
        tensorboard_callback
    )
else:
    model, hist = fit_dense(
        ds.train_scaled,
        ds.test_scaled,
        batch_size,
        nb_epochs,
        timestep,
        neurons_l1, neurons_l2, neurons_l3,
        tensorboard_callback
    )

# save the model files (.json and .h5) and the history logs of cost function
save = Model(model_dir, model_name)
save.saveModel(model)
cost_function(hist)

# print the last value of training loss for threshold reference
print(hist.history['val_loss'][-1:])