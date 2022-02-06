#-----------------------------------------------------------------
#-- Master Thesis - Model-Based Predictive Maintenance on FPGA
#--
#-- File : model_manipulation.py
#-- Description : Class for saving and loading trained model
#--
#-- Author : Sami Foery
#-- Master : MSE Mechatronics
#-- Date : 14.01.2022
#-----------------------------------------------------------------

from keras.models import model_from_json

class Model:

    def __init__(self, path, model_name):
        self.path = path
        self.model_name = model_name

    def saveModel(self, model):
        # serialize model to JSON
        model_json = model.to_json()
        with open(self.path + self.model_name + '.json', "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights(self.path + self.model_name + '.h5')
        print("Saved model to disk")

        return 0

    def loadModel(self):
        # load json and create model
        json_file = open(self.path + self.model_name + '.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        lstm_model = model_from_json(loaded_model_json)
        # load weights into new model
        lstm_model.load_weights(self.path + self.model_name + '.h5')
        print("Loaded model from disk")

        return lstm_model