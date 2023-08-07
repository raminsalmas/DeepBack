import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.special import softmax
from sklearn.preprocessing import minmax_scale
from scipy import stats
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers
from tensorflow.keras.callbacks import EarlyStopping

class MyModel:
    def __init__(self):
        self.path = "./Models/"
        self.run = "prediction"
        self.model = None

    def build_model(self):
        inputs_s1 = keras.Input(shape=(1), name="RFU")
        features = layers.Dense(1000, activation='relu', name="1000")(inputs_s1)
        features = layers.Dense(800, activation='relu', name="800")(features)
        features = layers.Dense(500, activation='relu', name="500")(features)
        features = layers.Dense(150, activation='relu', name="150")(features)
        features = layers.Dense(100, activation='relu', name="100")(features)
        features = layers.Dense(150, activation='relu', name="150_")(features)
        features = layers.Dense(500, activation='relu', name="500_")(features)
        features = layers.Dense(800, activation='relu', name="800_")(features)
        features = layers.Dense(1000, activation='relu', name="1000_")(features)
        features_2 = layers.Dense(1, name="FEX")(features)
        features_3 = layers.Dense(1, name="BEX")(features)
        self.model = keras.Model(inputs=inputs_s1, outputs=[features_2, features_3])

    def compile_model(self):
        opt = optimizers.Adam()
        self.model.compile(optimizer=opt, loss=["mean_squared_error", "mean_squared_error"])

    def train_model(self, X, FEX, BEX, Xv, FEX_v, BEX_v):
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=100)
        history = self.model.fit([X], [FEX, BEX], epochs=1000, batch_size=None, verbose=2,
                                 validation_data=([Xv], [FEX_v, BEX_v]))

        return history

    def evaluate_model(self, Xv, FEX_v, BEX_v):
        pp = self.model.evaluate([Xv], [FEX_v, BEX_v])
        return pp

    def save_model(self, filename):
        self.model.save(self.path + filename + '.h5')

    def predict(self, inputs):
        return self.model.predict(inputs.std(1))

    def calculate_metrics(self, predicted, expected):
        cor = r2_score(predicted, expected)
        err = mean_squared_error(predicted, expected)
        return cor, err

    def save_results(self, filename, cor, err):
        with open(self.path + filename, "a") as ddsd:
            ddsd.writelines("%.6f, %.6f" % (cor, err) + '\n')

    def plot_loss(self, history):
        plt.plot(history.history['loss'][10:], label="Training")
        plt.plot(history.history['val_loss'][10:], label="Validating")
        plt.title('Model Loss', size=8)
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(loc=0, ncol=2, frameon=False)
        plt.show()

    def plot_rfu(self, EXP, RFU_BF, error_, R):
        plt.plot(EXP, linewidth=1)
        RFU_BF_ = np.multiply(RFU_BF, -1)
        plt.plot(RFU_BF_, linewidth=1)
        plt.axhline(y=0., color='black', linestyle='-', linewidth=0.5)
        plt.title("MSE = %.4f; R2 = %.4f" % (error_, R), size=8)
        plt.show()

    def backexc(self, RFU, BEX, FEX):
        return (RFU - FEX) / (BEX - FEX)

    def process_inputs(self, Input_):
        RFU = Input_
        EXP = np.loadtxt("EXP.csv", delimiter=',')
        data = self.predict(Input_.var(1))
        FEX_f = data[0]
        BEX_f = data[1]
        RFU_BF = vec_backexc(RFU, np.repeat(BEX_f, RFU.shape[1], axis=1), np.repeat(FEX_f, RFU.shape[1], axis=1))
        return EXP, RFU_BF

def main():
    dataset = np.loadtxt("Data_training_fexbex_V2.csv", delimiter=',')
    validation_ = np.loadtxt("Validation_V1.csv", delimiter=',')

    Xv = validation_[:, 0:7].var(1)
    Yv = validation_[:, 7:9]
    FEX_v = (Yv[:, 0:1])
    BEX_v = (Yv[:, 1:2])

    np.random.shuffle(dataset)
    X = dataset[:, 0:7].var(1)
    Y = dataset[:, 7:9]
    FEX = (Y[:, 0:1])
    BEX = (Y[:, 1:2])

    my_model = MyModel()
    my_model.build_model()
    my_model.compile_model()

    history = my_model.train_model(X, FEX, BEX, Xv, FEX_v, BEX_v)
    pp = my_model.evaluate_model(Xv, FEX_v, BEX_v)

    my_model.plot_loss(history)

    cor, err = my_model.calculate_metrics(pp, history.history)
    my_model.save_results("cor_3.txt", cor, err)

    my_model.save_model("model_BK_FX_new_no_alpha_1_systems_Cor_%.6f_%.6f_%.6f_%.6f" % (cor, err, cor, cor))

    Input_ = np.loadtxt("Input.csv", delimiter=',', comments='#')
    EXP, RFU_BF = my_model.process_inputs(Input_)

    error_ = mean_absolute_error(EXP, RFU_BF)
    R = r2_score(EXP, RFU_BF)

    my_model.plot_rfu(EXP, RFU_BF, error_, R)
    
if __name__ == "__main__":
    main()
