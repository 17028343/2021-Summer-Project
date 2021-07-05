from __future__ import print_function
import keras
from keras.models import Sequential, Model, Input
from keras.layers import Dense, Dropout, Flatten, SpatialDropout2D, SpatialDropout1D, AlphaDropout
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
import seaborn as sns

# The X-vals of each data point for the gaussian distributions
X_Values = np.arange(0, 16, 1)

# allocating the size of the training data set
Training_Set_Size = 90000

# Randomly Setting Parameters to generate the different gaussians
np.random.seed(3)
Shift = np.random.uniform(low=0.0, high=15.0, size=Training_Set_Size) 
np.random.seed(2)
Peak_Height = np.random.uniform(low=0.0, high=5.0, size=Training_Set_Size)
np.random.seed(1)
Standard_Deviation = np.random.uniform(low=0.8, high=5.0, size=Training_Set_Size)

# Calculating the y-values for each gaussian in the training data set - X_Train is the training data inputs
np.random.seed(0)
x_train = np.transpose(Peak_Height*np.exp(-(X_Values[np.newaxis,].T-Shift)**2/(2*Standard_Deviation**2))+np.random.randn(len(X_Values),Training_Set_Size)*0.3)

# The true values of each distribution's shift, height and width - Y_Train is the training data targets
y_train=np.c_[Shift, Peak_Height, Standard_Deviation]


# allocating the size of the training data set
Validation_Set_Size = 10000

# Randomly Setting Parameters to generate the different gaussians
Vali_Shift = np.random.uniform(low=0.0, high=15.0, size=Validation_Set_Size) 
Vali_Peak_Height = np.random.uniform(low=0.0, high=5.0, size=Validation_Set_Size)
Vali_Standard_Deviation = np.random.uniform(low=0.8, high=5.0, size=Validation_Set_Size)

# Calculating the y-values for each gaussian in the training data set - X_Test is the validaiton dataset inputs
x_test = np.transpose(Vali_Peak_Height*np.exp(-(X_Values[np.newaxis,].T-Vali_Shift)**2/(2*Vali_Standard_Deviation**2))+np.random.randn(len(X_Values),Validation_Set_Size)*0.3)

# The true values of each distribution's shift, height and width - Y_Test is the validation dataset targets
y_test=np.c_[Vali_Shift, Vali_Peak_Height, Vali_Standard_Deviation]


batch_size=32

def get_dropout(input_tensor, p=0.5, mc=False):
    if mc:
        return Dropout(p)(input_tensor, training=True)
    else:
        return Dropout(p)(input_tensor)

def get_model(mc=False, act="relu"):
    inp = Input(16,)
    x = Dense(64, activation=act)(inp)
    x = get_dropout(x, p=0.05, mc=mc)
    x = Dense(10, activation=act)(x)
    x = get_dropout(x, p=0.05, mc=mc)
    out = Dense(3, activation='linear')(x)

    model = Model(inputs=inp, outputs=out)

    model.compile(loss='mean_squared_error',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


model = get_model(mc=False, act="relu")
mc_model = get_model(mc=True, act="relu")


earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
h = model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=500,
              verbose=1,
              validation_data=(x_test, y_test),
              callbacks=earlystop)

basicpredictions=model.predict(x_test)


# score of the normal model
score = model.evaluate(x_test, y_test, verbose=0)

# Plotting the Accuracy of the Model over time, both training and validation data
plt.plot(h.history['accuracy'])
plt.plot(h.history['val_accuracy'])
plt.title('Model Accuracy - no Dropout in Validation')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['Training Set', 'Validation Set'], loc='lower right')
plt.grid()
plt.show()

# Plotting the Loss of the Model over time, both training and validation data
plt.plot(h.history['loss'])
plt.plot(h.history['val_loss'])
plt.title('Model Loss - no Dropout in Validation')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['Training Set', 'Validation Set'], loc='upper right')
plt.grid()
plt.show()


print('Test loss:', score[0])
print('Test accuracy:', score[1])


h_mc = mc_model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=500,
              verbose=1,
              validation_data=(x_test, y_test),
              callbacks=earlystop)


# Plotting the Accuracy of the Model over time, both training and validation data
plt.plot(h_mc.history['accuracy'])
plt.plot(h_mc.history['val_accuracy'])
plt.title('Model Accuracy - Dropout in Validation')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['Training Set', 'Validation Set'], loc='lower right')
plt.grid()
plt.show()

# Plotting the Loss of the Model over time, both training and validation data
plt.plot(h_mc.history['loss'])
plt.plot(h_mc.history['val_loss'])
plt.title('Model Loss - Dropout in Validation')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['Training Set', 'Validation Set'], loc='upper right')
plt.grid()
plt.show()



import tqdm

mc_predictions = []
for i in tqdm.tqdm(range(500)):
    y_p = mc_model.predict(x_test, batch_size=1000)
    mc_predictions.append(y_p)

mc_predictions=np.array(mc_predictions, dtype="float32")

from scipy import stats

# Not sure which to go with, mean or mode?
montcarlpred=np.squeeze(np.array(stats.mode(mc_predictions, axis=0)[0]))
# montcarlpred=np.mean(mc_predictions, axis=0)

valcurveindex=0 # anywhere within [0, 9999]

sns.kdeplot(mc_predictions[:,valcurveindex,0], color="c")
plt.axvline(y_test[valcurveindex,0], color="k")
plt.axvline(basicpredictions[valcurveindex,0], color="violet")
plt.axvline(montcarlpred[valcurveindex,0], color="orange")
plt.legend(["Confidence KDEPlot", "True Value", "Regular Neural Net Prediction", "Monte Carlo Net Prediction"])
plt.xlabel('Shift Value')
plt.show()

sns.kdeplot(mc_predictions[:,valcurveindex,1], color="r")
plt.axvline(y_test[valcurveindex,1], color="k")
plt.axvline(basicpredictions[valcurveindex,1], color="violet")
plt.axvline(montcarlpred[valcurveindex,1], color="orange")
plt.legend(["Confidence KDEPlot", "True Value", "Regular Neural Net Prediction", "Monte Carlo Net Prediction"])
plt.xlabel('Height Value')
plt.show()

sns.kdeplot(mc_predictions[:,valcurveindex,2], color="lime")
plt.axvline(y_test[valcurveindex,2], color="k")
plt.axvline(basicpredictions[valcurveindex,2], color="violet")
plt.axvline(montcarlpred[valcurveindex,2], color="orange")
plt.legend(["Confidence KDEPlot", "True Value", "Regular Neural Net Prediction", "Monte Carlo Net Prediction"])
plt.xlabel('Standard Deviation Value')
plt.show()


Liner=np.linspace(0,15,1000)
plt.scatter(X_Values, x_test[valcurveindex,:], marker='o', c='lime', ec='k', lw=0.5, zorder=3)
plt.plot(Liner, 
            basicpredictions[valcurveindex,1] * np.exp(-(Liner-basicpredictions[valcurveindex,0])*(Liner-basicpredictions[valcurveindex,0])/(2*basicpredictions[valcurveindex,2]*basicpredictions[valcurveindex,2])),
            c='b', zorder=1)
plt.plot(Liner, 
            montcarlpred[valcurveindex,1] * np.exp(-(Liner-montcarlpred[valcurveindex,0])*(Liner-montcarlpred[valcurveindex,0])/(2*montcarlpred[valcurveindex,2]*montcarlpred[valcurveindex,2])),
            c='r', zorder=2)
plt.plot(Liner, 
            y_test[valcurveindex,1] * np.exp(-(Liner-y_test[valcurveindex,0])*(Liner-y_test[valcurveindex,0])/(2*y_test[valcurveindex,2]*y_test[valcurveindex,2])), 
            c='green', zorder=0)
plt.legend(['Basic Network Curve', 'Monte Carlo Network Curve', 'True Function', 'True Values'])
plt.show()


