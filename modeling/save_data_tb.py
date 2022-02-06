#-----------------------------------------------------------------
#-- Master Thesis - Model-Based Predictive Maintenance Tool FPGA
#--
#-- File : training.py
#-- Description : Saving results in .dat format for HLS4ML conversion
#--
#-- Author : Sami Foery
#-- Master : MSE Mechatronics
#-- Date : 14.01.2022
#-----------------------------------------------------------------

import prediction as pr
import pandas as pd

# input data save as .dat
output_df = pd.DataFrame({'X axis': pr.dataResume['Expected'].values})
output_df.to_csv('tb_data/input_data.dat', index=False)

# output reconstruction save data as .dat
output_df = pd.DataFrame({'X axis': pr.dataResume['Reconstructed'].values})
output_df.to_csv('tb_data/reconstruction.dat', index=False)
