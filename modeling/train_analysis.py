#-----------------------------------------------------------------
#-- Master Thesis - Model-Based Predictive Maintenance on FPGA
#--
#-- File : train_analysis.py
#-- Description : Analysis of training results and recording of useful information
#--
#-- Author : Sami Foery
#-- Master : MSE Mechatronics
#-- Date : 14.01.2022
#-----------------------------------------------------------------

import data_structure as ds
import pandas as pd
from matplotlib import pyplot as plt
from model_manipulation import Model
import seaborn as sns
sns.set(color_codes=True)
import numpy as np
import configparser

print("All libraries are loaded from train analysis")

# load configurations
parser = configparser.ConfigParser()
parser.read('network_config.ini')

# variables assignation
model_dir = parser.get('MODEL', 'MODEL_DIR')
model_name = parser.get('MODEL', 'MODEL_NAME')
batch_size = parser.getint('NETWORK', 'BATCH_SIZE')
Threshold = parser.get('DATA', 'THRESHOLD')

# load model from json and h5 files
model = Model(model_dir, model_name)
model = model.loadModel()

# reconstruct the entire training dataset to build up state for reconstruction and threshold definition
train_reshaped = ds.train_scaled[:, 0].reshape(len(ds.train_scaled), 1, 1)
print("Running reconstruction on train data")
trainAnalysis = model.predict(train_reshaped, batch_size=batch_size, verbose=1)
trainAnalysis = trainAnalysis.reshape(trainAnalysis.shape[0], trainAnalysis.shape[2])
trainAnalysis = pd.DataFrame(trainAnalysis, columns=ds.train.columns)
trainAnalysis['Expected'] = ds.train_scaled
trainAnalysis['Error'] = np.abs(trainAnalysis[0].values - trainAnalysis['Expected'].values) #CHANGE
trainAnalysis.index = ds.train.index

# plot L1 distribution (loss distribution)
scored = pd.DataFrame(index=ds.train.index)
Xtrain = ds.train_scaled.reshape(ds.train_scaled.shape[0], ds.train_scaled.shape[1])
scored['Loss_L1'] = trainAnalysis['Error']
plt.figure(figsize=(16, 9))
plt.title('L1 Loss Distribution', fontsize=16)
sns.distplot(scored['Loss_L1'], bins=20, kde=True, color='blue')
plt.xlim([0.0, .5])
plt.show()

# plot reconstructed vs expected values from train data
fig, ax = plt.subplots(2, 1, figsize=(16, 9), sharex=True)
ax[0].set_title("Reconstructed vs expected train data")
ax[0].set_ylabel('Acceleration values (g)')
ax[0].plot(trainAnalysis['Expected'].values, label='Expected')
ax[0].plot(trainAnalysis[0].values, label='Reconstructed')
ax[1].set_title("L1 reconstruction loss")
ax[1].set_xlabel('Time steps')
ax[1].set_ylabel('Loss values')
ax[1].plot(trainAnalysis['Error'].values, label='L1 error')
plt.show()

# save informations to use in ARM software processing
soft_data = pd.DataFrame(columns=["Values"], index=["Threshold", "Min train value", "Max train value"])
soft_data.loc["Threshold", "Values"] = Threshold
soft_data.loc["Min train value", "Values"] = np.min(ds.train.values)
soft_data.loc["Max train value", "Values"] = np.max(ds.train.values)
soft_data.to_csv('ARM_software_infos/training_info.txt', header=None)

