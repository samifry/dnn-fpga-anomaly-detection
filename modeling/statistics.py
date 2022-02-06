#-----------------------------------------------------------------
#-- Master Thesis - Model-Based Predictive Maintenance on FPGA
#--
#-- File : statistics.py
#-- Description : Statistical complement to the model analysis on the test data
#--
#-- Author : Sami Foery
#-- Master : MSE Mechatronics
#-- Date : 14.01.2022
#-----------------------------------------------------------------

import numpy as np
from matplotlib import pyplot as plt
from prediction import dataResume, window

print("All libraries are loaded from statistics")

# computation of statistics parameters
def compute_statistics(error):
	error = error.reshape(len(error), 1)
	mean = error.mean()
	cov = 0
	for e in error:
		cov += np.dot((e - mean).reshape(len(e), 1), (e - mean).reshape(1, len(e)))
	cov /= len(error)

	return error, mean, cov

# computation of the mahalanobis distance
def compute_mahalanobis(x, mean, cov):
	d = np.dot(x-mean, np.linalg.inv(cov))
	d = np.dot(d, (x-mean).T)
	return d

# list and save mahalanobis distance values
def list_mahalanobis(error, mean, cov):
	m_dist = []
	for e in error:
		m_dist.append(compute_mahalanobis(e, mean, cov))

	return m_dist

# plot the mahalanobis distance
def plotMahalanobisDistance(m_dist, window):
	plt.figure(figsize=(16, 9))
	plt.style.use('ggplot')
	plt.plot(m_dist, color='r', label='Mahalanobis Distance')
	plt.title("Mahalanobis distance study")
	plt.xlabel('Time steps')
	plt.ylabel('Mahalanobis Distance')
	plt.xlim(-10, window + 10)
	plt.ylim(0, max(m_dist) + 5)

	plt.legend(fontsize=15)
	plt.show()

# compute statistics paramaters
error, mean, cov = compute_statistics(dataResume['Error'].values)

# mahalanobis distance list values
m_dist = list_mahalanobis(error, mean, cov)

# plot mahalanobis distance values
plotMahalanobisDistance(m_dist, window)