#-----------------------------------------------------------------
#-- Master Thesis - Model-Based Predictive Maintenance on FPGA
#--
#-- File : data_structure.py
#-- Description : Split and transformation of acquired data for training and testing
#--
#-- Author : Sami Foery
#-- Master : MSE Mechatronics
#-- Date : 14.01.2022
#-----------------------------------------------------------------

import os
import pandas as pd
import configparser
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
sns.set(color_codes=True)
from matplotlib import pyplot as plt
import tensorflow as tf
from numpy.random import seed

print("All libraries are loaded from data structure")

# set random seed
seed(10)
tf.random.set_seed(10)

# load and merge sensor samples
def importData(data_dir):
    merge_data = pd.DataFrame()
    for filename in sorted(os.listdir(data_dir), key=len):
        dataset = pd.read_csv(os.path.join(data_dir, filename), usecols=[0], header=None) #CHANGE
        dataset.index = np.arange(len(merge_data), len(dataset) + len(merge_data))
        merge_data = merge_data.append(dataset)

    return merge_data

# split global dataset to train and test datasets
def splitData(merge_data, trainRate):
   train = merge_data[0:int(len(merge_data)*trainRate)]
   test = merge_data[len(train):]

   return train, test

# scale train and test data to [0, 1]
def scale(train, test):
  # fit scaler
  scaler = MinMaxScaler(feature_range=(0, 1))
  scaler = scaler.fit(train)
  # transform train
  train_scaled = scaler.transform(train)
  train_scaled = train_scaled.astype('float16')
  # transform test
  test_scaled = scaler.transform(test)
  test_scaled = test_scaled.astype('float16')
  return scaler, train_scaled, test_scaled

# inverse scaling for a forecasted value
def invert_scale(scaler, value):
  array = np.array(value)
  array = array.reshape(1, len(array))
  inverted = scaler.inverse_transform(array)
  return inverted[0, -1]

# plot train and test data
def plotTrainTestData(train_data, test_data):
  plt.figure(figsize=(16, 9))
  plt.subplot(211)
  plt.plot(train_data)
  plt.ylabel('Acceleration values (g)')
  plt.title('Train and test data')
  plt.subplot(212)
  plt.plot(test_data)
  plt.xlabel('Time steps')
  plt.ylabel('Acceleration values (g)')
  plt.show()

  return 0

# load configurations
parser = configparser.ConfigParser()
parser.read('network_config.ini')

# variables assignation
data_dir = parser.get('DATA', 'DATA_DIR')
trainRate = parser.getfloat('DATA', 'TRAIN_RATE')

# run data structure processing
all_data = importData(data_dir)
train, test = splitData(all_data, trainRate)
scaler, train_scaled, test_scaled = scale(train, test)
plotData = plotTrainTestData(train_scaled, test_scaled)
print(plotData)