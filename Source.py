#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline 
import numpy as np
from numpy import loadtxt
from keras.models import Sequential, model_from_json, load_model
from keras.layers import Dense, Dropout
from keras import metrics, regularizers
import tensorflow as tf
from scipy.special import softmax
from sklearn.preprocessing import minmax_scale 
#from keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import EarlyStopping
from keras.constraints import unit_norm
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy import stats
from keras import backend as K
from tensorflow.keras import optimizers
from tensorflow import keras
from keras import layers
import pickle





path = "./Models/"

run = "predictio"

dataset = np.loadtxt("Data_training_fexbex_V2.csv", delimiter=',')
validation_ = np.loadtxt("Validation_V1.csv", delimiter=',')
Xv = validation_[:,0:7].var(1)
#Xv = validation_[:,0:7]
Yv = validation_[:,7:9]

FEX_v = (Yv[:,0:1])
BEX_v = (Yv[:,1:2])



np.random.shuffle(dataset)

X = dataset[:,0:7].var(1)
#X = minmax_scale(dataset[:,0:7], axis=1, feature_range=(0, 0.1)).std(1)
Y = dataset[:,7:9]

FEX = (Y[:,0:1])
BEX = (Y[:,1:2])


inputs_s1 = keras.Input(shape=(1), name="RFU")
#output_s1 = keras.Input(output_1.shape[1])
#output_s2 = keras.Input(shape=(output_2.shape[1]))


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

model = keras.Model(inputs=inputs_s1, outputs=[features_2,features_3 ])

print(model.summary())
print(keras.utils.plot_model(model, "ff_V1.png", dpi=300,
                             show_shapes=True, show_layer_names=True))

opt=optimizers.Adam()
model.compile(optimizer=opt, loss=["mean_squared_error", "mean_squared_error"],)
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=100)
history =model.fit([X],[FEX, BEX], epochs=1000, batch_size=None, verbose=2,
        validation_data=([Xv], [FEX_v, BEX_v]),   ) #callbacks=[es]

pp = model.evaluate([Xv],[FEX_v, BEX_v])
o = history.history


plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=None)
plt.subplot(2,1,1)
plt.plot(history.history['loss'][10:], label="Training")
plt.plot(history.history['val_loss'][10:], label="Validating")

plt.title('Model Loss', size=8)
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend( loc=0, ncol=2, frameon=False)

plt.savefig(path + "%.6f.png" %(R), facecolor='w',dpi=300, edgecolor='w',
        orientation='portrait', papertype='legal', format="png",
        transparent=None, bbox_inches="tight", frameon=None,)
with open(path + "%.6f_history" %(R), 'wb') as file_pi:
    pickle.dump(history.history, file_pi)
plt.show()





# =============================================================================
# Validation of the model
# 
# =============================================================================
if run== "prediction":
        
        
        
        Input_ = np.loadtxt("Input.csv", delimiter=',', comments='#')
        
        try:
            row, col = Input_.shape
        except ValueError:
            row=1; col = len(Input_)
            
       
        Out= model.predict(Input_.std(1))
        
        print(Out)
        exp = np.loadtxt("try_pre.csv", delimiter=",")
        cor = r2_score(Out, exp)
        print(exp)
        print(Out)
        
        err = mean_squared_error(Out, exp)
        slope, intercept, r_fex, p_fex, std_err = stats.linregress(Out[:, 0:1].flatten(), exp[:, 0:1].flatten())
        slope, intercept, r_bex, p_bex, std_err = stats.linregress(Out[:, 1:2].flatten(), exp[:, 1:2].flatten())
        r_bex = r_bex**2
        r_fex = r_fex**2
        
        fi = "cor_3.txt"
        with open(path + fi, "a") as ddsd:
            ddsd.writelines("%.6f, %.6f,%.6f,%.6f" %(cor, err, r_fex, r_bex,   ) + '\n')
        
        model.save(path + "model_BK_FX_new_no_alpha_1_systems_Cor_%.6f_%.6f_%.6f_%.6f.h5" % (cor, err, r_fex,r_bex))
        
        np.savetxt(path + "Output_BK_FX_new_no_alpha_1_systems_Cor_%.6f_%.6f_%.6f_%.6f.csv" %(cor, err, r_fex, r_bex), Out )

        

        


# =============================================================================
#         Validation with RFU values
# =============================================================================



def backexc(RFU, BEX, FEX):
    return (RFU - FEX)/(BEX - FEX)

vec_backexc=np.vectorize(backexc)


Input_ = np.loadtxt("Input.csv", delimiter=',', comments='#')



RFU=Input_
EXP = np.loadtxt("EXP.csv", delimiter=',')
Out= model.predict(Input_.var(1))

data = Out

FEX_f = data[0]
BEX_f = data[1]

RFU_BF = vec_backexc(RFU, np.repeat(BEX_f, RFU.shape[1], axis=1) ,np.repeat(FEX_f, RFU.shape[1], axis=1))


plt.subplot(2,1,1)



plt.plot(EXP, linewidth=1, )
#print(RFU_BF)
RFU_BF_ = np.multiply(RFU_BF, -1)
RFU_ = np.multiply(RFU, -1)
plt.plot(RFU_BF_, linewidth=1)
plt.axhline(y=0., color='black', linestyle='-', linewidth=0.5)
error_ = mean_absolute_error(EXP, RFU_BF)
R = r2_score(EXP, RFU_BF)
#print(error_)
plt.title("MSE = %.4f; R2 = %.4f" %(error_, R), size=8)


    
model.save(path + "model_FAPI_%.6f_%.06f.h5" % (R, error_))
plt.savefig(path + "RFU_%.6f.png" %(error_), facecolor='w',dpi=300, edgecolor='w',
      orientation='portrait', papertype='legal', format="png",
      transparent=None, bbox_inches="tight", frameon=None,)
    
plt.show()
    
 
