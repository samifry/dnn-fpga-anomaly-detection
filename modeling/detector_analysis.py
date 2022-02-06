#-----------------------------------------------------------------
#-- Master Thesis - Model-Based Predictive Maintenance on FPGA
#--
#-- File : detector_analysis.py
#-- Description : Structure and files preparation for NAB analysis
#--
#-- Author : Sami Foery
#-- Master : MSE Mechatronics
#-- Date : 14.01.2022
#-----------------------------------------------------------------

import data_structure as ds
import pandas as pd
import json
import numpy as np
import configparser
from model_manipulation import Model

print("All libraries from detector analysis are loaded")
print("Preparation of CSV files for NAB analysis")

# save inputs in NAB format frame
def getInputs(inputs, timestamp):
    inputsData = pd.DataFrame(index=np.arange(len(inputs)), columns=['timestamp', 'value'])

    for i in range(len(inputs)):
        inputsData['timestamp'].iloc[i] = timestamp.iloc[i]
        input = inputs[i]
        input = ds.invert_scale(ds.scaler, input)
        inputsData['value'].iloc[i] = input

    return inputsData

# running prediction using NAB data and save results in NAB format frame
def algoRunning(model, data, timestep, features, timestamp):
    reconstruction = pd.DataFrame(index=np.arange(len(data)), columns=['timestamp', 'value'])
    error = list()
    reconstruction['timestamp'] = timestamp
    for i in range(len(data)):
        X = data[i]
        X = X.reshape(X.shape[0], timestep, features)
        yhat = model.predict(X, batch_size=1)
        yhat = yhat[0,0]
        gap = np.abs(yhat - X)
        yhat = ds.invert_scale(ds.scaler, yhat)
        reconstruction['value'].iloc[i] = yhat
        error.append(gap[0,0])
        print('Loop' + str(i), 'Norm error =' + str(gap[0,0]))

    return reconstruction, error

# anomaly classification using detection threshold
def getAnomalyScore(error, threshold):
    anomalyBoard = pd.DataFrame(index=np.arange(len(error)))
    anomalyBoard['Error'] = error
    anomalyBoard['Threshold'] = threshold
    anomalyBoard['anomaly_score'] = anomalyBoard['Error'] > anomalyBoard['Threshold']
    anomaly_score = anomalyBoard['anomaly_score'].values
    anomaly_score = anomaly_score.reshape(-1, 1)

    return anomaly_score

# add real anomalies in the frame (reference in NAB/labels/combined_windows.json)
def getLabel(timestamp, timestampLabel):

    label = (timestamp >= timestampLabel[0][0]) & (timestamp <= timestampLabel[0][1]) |\
        (timestamp >= timestampLabel[1][0]) & (timestamp <= timestampLabel[1][1]) |\
        ((timestamp >= timestampLabel[2][0]) & (timestamp <= timestampLabel[2][1])) |\
        ((timestamp >= timestampLabel[3][0]) & (timestamp <= timestampLabel[3][1]))

    return label

# generate csv file in NAB format
def getCSV(inputs, results):
    inputs.to_csv(gen_nab_dir + nab_inputs, index=False)
    results.to_csv(gen_nab_dir + nab_results, index=False)

    return 0

# load configurations
parser = configparser.ConfigParser()
parser.read('network_config.ini')

# variables assignation
model_dir = parser.get('MODEL', 'MODEL_DIR')
model_name = parser.get('MODEL', 'MODEL_NAME')
timestep = parser.getint('DATA', 'TIMESTEP')
features = parser.getint('DATA', 'FEATURES')
threshold = parser.getfloat('DATA', 'THRESHOLD')
gen_nab_dir = parser.get('DATA', 'DATA_GEN_NAB_DIR')
nab_inputs = parser.get('DATA', 'NAB_INPUTS')
nab_results = parser.get('DATA', 'NAB_RESULTS')
data_dir = parser.get('DATA', 'DATA_DIR')

# load model from json and h5 files
model = Model(model_dir, model_name)
model = model.loadModel()

# load json file for labeling part
with open('dataset/NAB_real_anomalies/combined_windows.json') as f:
   labelWindow = json.load(f)
timestampLabel = labelWindow["realKnownCause/machine_temperature_system_failure.csv"]

# get inputs from nab data
allValues = np.concatenate((ds.train_scaled, ds.test_scaled))
timestamp = pd.read_csv(data_dir + '/machine_temperature_system_failure.csv', usecols=[0])
inputsNAB = getInputs(allValues, timestamp)
# running model with nab data
results, error = algoRunning(model, allValues, timestep, features, timestamp)
# get binary anomaly scores (anomalies = 1, no-anomalies = 0)
anomalyScore = getAnomalyScore(np.array(error), threshold)
results['anomaly_score'] = anomalyScore
results['anomaly_score'] = results['anomaly_score'].astype(int)
# get label values (real anomalies datetime windows)
label = getLabel(timestamp, timestampLabel)
results['label'] = label
results['label'] = results['label'].astype(int)
convertCSV = getCSV(inputsNAB, results)

print("CSV inputs and results files are ready")