#-----------------------------------------------------------------
#-- Master Thesis - Model-Based Predictive Maintenance on FPGA
#--
#-- File : plot_results.py
#-- Description : Simulation and hardware levels results plotting
#--
#-- Author : Sami Foery
#-- Master : MSE Mechatronics
#-- Date : 03.02.2022
#-----------------------------------------------------------------

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.stats import norm
import statistics

# load files and build data frames
def loadFiles():
    expected = pd.read_csv('tb_data/tb_input_features.dat')
    reconstruction = pd.read_csv('tb_data/tb_output_predictions.dat')
    csim = pd.read_csv('tb_data/csim_results.log')
    hw = pd.read_csv('tb_data/logs.dat')
    real_data = pd.read_csv('tb_data/data_training.dat')
    real_test = pd.read_csv('tb_data/data_test.dat')
    spi_data = pd.read_csv('tb_data/normalized_data.dat')
    spi_reconstructed = pd.read_csv('tb_data/reconstructed_data.dat')
    dfCompare = pd.DataFrame(
        index=np.arange(len(reconstruction.values)),
        columns=['Expected',
                 'Reconstruction',
                 'Csim',
                 'hw',
                 'Conversion error',
                 'Threshold',
                 'Validation',
                 'Reconstruction error'
                 ]
    )
    dfCompare['Expected'] = expected
    dfCompare['Reconstruction'] = reconstruction
    dfCompare['Csim'] = csim
    dfCompare['hw'] = hw

    dfRealData = pd.DataFrame(
        index=np.arange(len(real_data.values)),
        columns=['X Axis Training',
                 'X Axis Test']
    )
    dfRealData['X Axis Training'] = real_data
    dfRealData['X Axis Test'] = real_test

    dfSPI = pd.DataFrame(
        index=np.arange(len(spi_data.values)),
        columns=['X Axis Real Data',
                 'X Axis Reconstructed Data',
                 'X Axis Reconstruction Error',
                 'Threshold']
    )
    dfSPI['X Axis Real Data'] = spi_data
    dfSPI['X Axis Reconstructed Data'] = spi_reconstructed
    dfSPI['X Axis Reconstruction Error'] = np.abs(dfSPI['X Axis Real Data'].values - dfSPI['X Axis Reconstructed Data'].values)
    dfSPI['Threshold'] = 0.068
    dfSPI['Anomaly'] = dfSPI['X Axis Reconstruction Error'] > dfSPI['Threshold']

    return dfCompare, dfRealData, dfSPI

# plot C simulation results for comparison
def conversionAnalysis(table):
    table['Conversion error'] = np.abs(table['Reconstruction'].values - table['hw'].values)
    table['Threshold'] = np.abs(max(table['Conversion error'].values))
    table['Validation'] = table['Conversion error'].values < table['Threshold'].values
    table['Reconstruction error'] = np.abs(table['Expected'].values - table['hw'].values)

    return table

# add plot hardware level results
def showResults(fullFrame):
    fig, ax = plt.subplots(3, 1, figsize=(16, 6), sharex=True)
    ax[0].set_title("Comparison between expected, reconstructed and converted values")
    ax[0].set_ylabel("Values")
    ax[0].plot(fullFrame['Expected'].values)
    ax[0].plot(fullFrame['Reconstruction'].values)
    ax[0].plot(fullFrame['Csim'].values)
    ax[0].plot(fullFrame['hw'].values)
    ax[0].legend(['Expected', 'Reconstructed', 'HW sim results', 'HW results'])
    ax[1].set_title("Conversion error")
    ax[1].set_ylabel("Values")
    ax[1].plot(fullFrame['Conversion error'].values)
    ax[2].set_title("Reconstruction error in HW level")
    ax[2].set_xlabel("Index")
    ax[2].set_ylabel("Error values")
    ax[2].plot(fullFrame['Reconstruction error'].values)
    plt.show()

    return None

# plot acquired data
def acquisitionDATA(dataFrame):
    plt.figure(figsize=(16, 6))
    plt.title("SPI X axis data acquisition")
    plt.ylabel("Acceleration values (g)")
    plt.plot(dataFrame["X Axis Training"].values)
    plt.plot(dataFrame["X Axis Test"].values)
    plt.legend(['Real train data', 'Real test data'])
    plt.show()

    return None

# plot results from realtime test logs
def showReal(SPIFrame):
    fig, ax = plt.subplots(3, 1, figsize=(16, 6), sharex=True)
    ax[0].set_title("X axis real and reconstructed data comparison")
    ax[0].set_ylabel("Acceleration (g)")
    ax[0].plot(SPIFrame['X Axis Real Data'].values)
    ax[0].plot(SPIFrame['X Axis Reconstructed Data'].values)
    ax[0].legend(['Expected', 'Reconstructed'], loc="lower right")
    ax[1].set_title("X axis reconstruction error")
    ax[1].set_ylabel("Absolute error")
    ax[1].plot(SPIFrame['X Axis Reconstruction Error'].values)
    ax[1].plot(SPIFrame['Threshold'].values, color="red")
    ax[1].legend(["Error", "Threshold"], loc="lower right")
    ax[2].set_title("Anomalies identification")
    ax[2].set_xlabel("Time steps")
    ax[2].set_ylabel("Acceleration (g)")
    ax[2].plot(SPIFrame['X Axis Real Data'].values)

    # plot anomaly in testing data
    anomalies = SPIFrame[SPIFrame.Anomaly == True]
    anomalies.head()
    SPIFrame['X Axis Real Data'].index = np.arange(0, len(SPIFrame['X Axis Real Data']))

    sns.scatterplot(
        x=anomalies.index,
        y=SPIFrame['X Axis Real Data'].iloc[anomalies.index],
        color=sns.color_palette()[3],
        s=52,
        label="Anomaly"
    )
    ax[2].legend(loc='lower right')
    plt.show()

    # normal distribution on train reconstruction error
    error = SPIFrame['X Axis Reconstruction Error'].values
    mean = statistics.mean(error)
    sd = statistics.stdev(error)
    plt.plot(error, norm.pdf(error, mean, sd))
    plt.show()

    return None

loadTable, loadRealData, loadSPI  = loadFiles()
fullTable = conversionAnalysis(loadTable)
showResults(fullTable)
acquisitionDATA(loadRealData)
showReal(loadSPI)
