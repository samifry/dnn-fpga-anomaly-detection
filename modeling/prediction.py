#-----------------------------------------------------------------
#-- Master Thesis - Model-Based Predictive Maintenance on FPGA
#--
#-- File : prediction.py
#-- Description : Model analysis module on test data
#--
#-- Author : Sami Foery
#-- Master : MSE Mechatronics
#-- Date : 14.01.2022
#-----------------------------------------------------------------
import data_structure as ds
import pandas as pd
import configparser
from matplotlib import pyplot as plt
from pandas import DataFrame
from model_manipulation import Model
import numpy as np
import seaborn as sns

print("All libraries are loaded from prediction")

# prediction and create dataframe for storing information
def dataStoring(model, data_scaled, window, timestep, features, batch_size, one_step):
	dfAnomaly = pd.DataFrame(index=np.arange(window), columns=['Expected', 'Reconstructed', 'Error'], dtype=np.float64)

	# prediction using one input after the other (batch size of 1)
	if one_step:
		for i in range(window):

			# make one-step forecast
			X = data_scaled[i]
			X = X.reshape(X.shape[0], timestep, features)
			yhat = model.predict(X, batch_size=batch_size)
			yhat = yhat[0,0]

			# store forecast
			dfAnomaly['Reconstructed'].iloc[i] = yhat
			expected = data_scaled[i]
			dfAnomaly['Expected'].iloc[i] = data_scaled[i]
			print('DataNum=%d, Reconstructed=%f, Expected=%f' % (i+1, yhat, expected))

			# report performance)
			test_mae_loss = np.mean(np.abs(yhat-expected))
			print('Test L1 loss: %.3f' % test_mae_loss)
			dfAnomaly['Error'].iloc[i] = test_mae_loss

	# prediction using several inputs at the same time (batch size greather than 1)
	else:
		X = data_scaled[0:window]
		X = X.reshape(X.shape[0], timestep, features)
		print("Running reconstruction of test data")
		yhat = model.predict(X, batch_size=batch_size, verbose=1)
		yhat = yhat.reshape(window, 1)
		dfAnomaly['Reconstructed'] = yhat
		dfAnomaly['Expected'] = data_scaled[0:window]
		dfAnomaly['Error'] = np.abs(dfAnomaly['Reconstructed'].values - dfAnomaly['Expected'].values)

	# looking at the summary of the model
	model.summary()

	return dfAnomaly

# plots of expected vs reconstructed/predicted values
def showResults(dfAnomaly, window, threshold):
	plt.figure(figsize=(16, 9))
	plt.plot(dfAnomaly['Expected'])
	plt.plot(dfAnomaly['Reconstructed'])
	plt.ylabel('Acceleration values (g)')
	plt.xlabel('Time steps')
	plt.legend(['Target', 'Reconstruction'])
	plt.title('Reconstruction of acceleration values')
	plt.show()

	# summarize results
	results = DataFrame()
	results['L1 Loss'] = dfAnomaly['Error'].values
	print(results.describe())
	plt.show()

	# calculate the l1 loss (norm loss)
	scored = pd.DataFrame(index=np.arange(0, window))
	scored['L1 Loss'] = dfAnomaly['Error']
	scored['Threshold'] = threshold
	scored['Anomaly'] = scored['L1 Loss'] > scored['Threshold']
	# plot mae for anomaly detection
	scored.plot(logy=True, figsize=(16, 9), color=['blue', 'red'])
	plt.title("Absolute errors of reconstructed and expected values")
	plt.xlabel("Time steps")
	plt.ylabel("Absolute errors")
	plt.show()

	# plot anomalies in testing data
	anomalies = scored[scored.Anomaly == True]
	anomalies.head()
	ds.test.index = np.arange(0, len(ds.test))
	plt.figure(figsize=(16, 9))
	plt.title("Anomalies identification in real data")
	plt.xlabel("Time steps")
	plt.ylabel("Acceleration values (g)")
	plt.plot(
		ds.test[:window].index,
		ds.test[:window],
		label='Acceleration values (g)'
	)

	sns.scatterplot(
		x=anomalies.index,
		y=ds.test.loc[anomalies.index, 0],
		color=sns.color_palette()[3],
		s=52,
		label='anomaly'
	)

	plt.xticks(rotation=25)
	plt.legend()
	plt.show()

	return scored

# load configurations
parser = configparser.ConfigParser()
parser.read('network_config.ini')

# variables assignation
model_dir = parser.get('MODEL', 'MODEL_DIR')
model_name = parser.get('MODEL', 'MODEL_NAME')
timestep = parser.getint('DATA', 'TIMESTEP')
features = parser.getint('DATA', 'FEATURES')
threshold = parser.getfloat('DATA', 'THRESHOLD')
batch_size = parser.getint('NETWORK', 'BATCH_SIZE')
one_step = parser.getboolean('NETWORK', 'ONE_STEP')
window = len(ds.test_scaled)

# load model from json and h5 files
model = Model(model_dir, model_name)
model = model.loadModel()

# storing data and results in a dataframe
dataResume = dataStoring(
	model,
	ds.test_scaled,
	window,
	timestep,
	features,
	batch_size,
	one_step
)

results = showResults(dataResume, window, threshold)