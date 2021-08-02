import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt
import keras
from keras.models import Model, Input
from keras.layers import Dense, Dropout


# The X-vals of each data point for the gaussian distributions
X_Values = np.array([1074.5, 1074.62, 1074.74])

# allocating the size of the training data set
Training_Set_Size = 90000

# Randomly Setting Parameters to generate the different gaussians
np.random.seed(0)
Shift = np.random.normal(-0.019, 0.01, size=Training_Set_Size)+1074.62
np.random.seed(1)
Peak_Height = np.random.uniform(low=1750, high=21875, size=Training_Set_Size)
np.random.seed(2)
Standard_Deviation = np.random.exponential(scale=0.0358, size=Training_Set_Size)+0.14

# Calculating the y-values for each gaussian in the training data set - X_Train is the training data inputs
Noiseless_A=Peak_Height * np.exp(-(X_Values[0]-Shift)*(X_Values[0]-Shift)/(2*Standard_Deviation*Standard_Deviation))
Noiseless_B=Peak_Height * np.exp(-(X_Values[1]-Shift)*(X_Values[1]-Shift)/(2*Standard_Deviation*Standard_Deviation))
Noiseless_C=Peak_Height * np.exp(-(X_Values[2]-Shift)*(X_Values[2]-Shift)/(2*Standard_Deviation*Standard_Deviation))

np.random.seed(3)
Poisson_Noise_A=np.random.poisson(lam=Noiseless_A)
np.random.seed(4)
Poisson_Noise_B=np.random.poisson(lam=Noiseless_B)
np.random.seed(5)
Poisson_Noise_C=np.random.poisson(lam=Noiseless_C)

Poisson_Noise=np.c_[Poisson_Noise_A, Poisson_Noise_B, Poisson_Noise_C]

np.random.seed(6)
Gaussian_Noise=np.random.normal(0, 140, size=(Training_Set_Size, 3))

# These are the Noiseless Y values
x_train = np.transpose(Peak_Height*np.exp(-(X_Values[np.newaxis,].T-Shift)**2/(2*Standard_Deviation**2)))

# Adding the noise to the generated Y values
for i in np.arange(0,3,1):
    for j in np.arange(0,Training_Set_Size,1):
        x_train[j,i]=+np.sqrt(Poisson_Noise[j,i]**2+Gaussian_Noise[j,i]**2)
        

x_train_scaled=np.array(x_train, dtype="float32")

for i in np.arange(0,90000,1):
    x_train_scaled[i,0]=x_train[i,0]/np.max(x_train, axis=1)[i]
    x_train_scaled[i,1]=x_train[i,1]/np.max(x_train, axis=1)[i]
    x_train_scaled[i,2]=x_train[i,2]/np.max(x_train, axis=1)[i]

# The true values of each distribution's shift, height and width - Y_Train is the training data targets
y_train=np.c_[Shift, Peak_Height, Standard_Deviation]


y_train_scaled=np.array(y_train, dtype="float32")

for i in np.arange(0,90000,1):
    y_train_scaled[i,1]=y_train[i,1]/np.max(x_train, axis=1)[i]


# allocating the size of the testing data set
Test_Set_Size = 10000

# Randomly Setting Parameters to generate the different gaussians
np.random.seed(7)
Shift_Test = np.random.normal(-0.019, 0.01, size=Test_Set_Size)+1074.62
np.random.seed(8)
Peak_Height_Test = np.random.uniform(low=1750, high=21875, size=Test_Set_Size)
np.random.seed(9)
Standard_Deviation_Test = np.random.exponential(scale=0.0358, size=Test_Set_Size)+0.14

# Calculating the y-values for each gaussian in the training data set - X_Train is the training data inputs
Noiseless_A_Test=Peak_Height_Test * np.exp(-(X_Values[0]-Shift_Test)*(X_Values[0]-Shift_Test)/(2*Standard_Deviation_Test*Standard_Deviation_Test))
Noiseless_B_Test=Peak_Height_Test * np.exp(-(X_Values[1]-Shift_Test)*(X_Values[1]-Shift_Test)/(2*Standard_Deviation_Test*Standard_Deviation_Test))
Noiseless_C_Test=Peak_Height_Test * np.exp(-(X_Values[2]-Shift_Test)*(X_Values[2]-Shift_Test)/(2*Standard_Deviation_Test*Standard_Deviation_Test))

np.random.seed(10)
Poisson_Noise_A_Test=np.random.poisson(lam=Noiseless_A_Test)
np.random.seed(11)
Poisson_Noise_B_Test=np.random.poisson(lam=Noiseless_B_Test)
np.random.seed(12)
Poisson_Noise_C_Test=np.random.poisson(lam=Noiseless_C_Test)

Poisson_Noise_Test=np.c_[Poisson_Noise_A_Test, Poisson_Noise_B_Test, Poisson_Noise_C_Test]

np.random.seed(13)
Gaussian_Noise_Test=np.random.normal(0, 140, size=(Test_Set_Size, 3))

# These are the Noiseless Y values
x_test = np.transpose(Peak_Height_Test*np.exp(-(X_Values[np.newaxis,].T-Shift_Test)**2/(2*Standard_Deviation_Test**2)))

# Adding the noise to the generated Y values
for i in np.arange(0,3,1):
    for j in np.arange(0,Test_Set_Size,1):
        x_test[j,i]=+np.sqrt(Poisson_Noise_Test[j,i]**2+Gaussian_Noise_Test[j,i]**2)
        

x_test_scaled=np.array(x_test, dtype="float32")

for i in np.arange(0,10000,1):
    x_test_scaled[i,0]=x_test[i,0]/np.max(x_test, axis=1)[i]
    x_test_scaled[i,1]=x_test[i,1]/np.max(x_test, axis=1)[i]
    x_test_scaled[i,2]=x_test[i,2]/np.max(x_test, axis=1)[i]
        

# The true values of each distribution's shift, height and width - Y_test is the testing data targets
y_test=np.c_[Shift_Test, Peak_Height_Test, Standard_Deviation_Test]

y_test_scaled=np.array(y_test, dtype="float32")

for i in np.arange(0,10000,1):
    y_test_scaled[i,1]=y_test[i,1]/np.max(x_test, axis=1)[i]


# Showing the Train and Test Gaussians
GaussianNumber=421
Liner=np.linspace(1074.4,1074.8,1000)
plt.scatter(X_Values, x_train[GaussianNumber,:], s=20, edgecolors=("k"), c='lime')
plt.plot(Liner, 
            Peak_Height[GaussianNumber] * np.exp(-(Liner-Shift[GaussianNumber])*(Liner-Shift[GaussianNumber])/(2*Standard_Deviation[GaussianNumber]*Standard_Deviation[GaussianNumber])),
            c='g')
plt.scatter(X_Values, x_test[GaussianNumber,:], s=20, edgecolors=("k"), c='cyan')
plt.plot(Liner, 
            Peak_Height_Test[GaussianNumber] * np.exp(-(Liner-Shift_Test[GaussianNumber])*(Liner-Shift_Test[GaussianNumber])/(2*Standard_Deviation_Test[GaussianNumber]*Standard_Deviation_Test[GaussianNumber])),
            c='b')
plt.show()


def get_dropout(input_tensor, p=0.5, mc=False):
    if mc:
        return Dropout(p)(input_tensor, training=True)
    else:
        return Dropout(p)(input_tensor)

def get_model(mc=False, act="relu"):
    inp = Input(3,)
    x = Dense(256, activation=act)(inp)
    x = get_dropout(x, p=0.05, mc=mc)
    x = Dense(128, activation=act)(x)
    x = get_dropout(x, p=0.05, mc=mc)
    x = Dense(64, activation=act)(x)
    x = get_dropout(x, p=0.05, mc=mc)
    x = Dense(32, activation=act)(x)
    x = get_dropout(x, p=0.05, mc=mc)
    x = Dense(10, activation=act)(x)
    x = get_dropout(x, p=0.05, mc=mc)
    out = Dense(3, activation='linear')(x)

    model = Model(inputs=inp, outputs=out)

    model.compile(loss='mean_squared_error',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

mc_model = get_model(mc=True, act="selu")

earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
batch_size=120

h_mc = mc_model.fit(x_train_scaled, y_train_scaled,
              batch_size=batch_size,
              epochs=200,
              verbose=1,
              validation_data=(x_test_scaled, y_test_scaled),
              callbacks=earlystop,
                    )

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

# Monte Carlo Network Predictions
mc_predictions = []
for i in tqdm.tqdm(range(500)):
    y_p = mc_model.predict(x_test_scaled, batch_size=1000)
    mc_predictions.append(y_p)

mc_predictions=np.array(mc_predictions, dtype="float32")
montcarlpred=np.median(mc_predictions, axis=0)


for i in np.arange(0,10000,1):
    montcarlpred[i,1]=montcarlpred[i,1]*np.max(x_test, axis=1)[i]

# Implementing the Analytical solutions to the gaussians
a = np.log(x_test[:,2]/x_test[:,1])
b = np.log(x_test[:,0]/x_test[:,1])
d = 0.12

Analytic_Width = np.sqrt((-2*d**2)/(a+b))
Analytic_Shift = (a-b)*Analytic_Width**2/(4*d)
Analytic_Height = x_test[:,1]*np.exp(Analytic_Shift**2/Analytic_Width**2)
Analytic_Shift = Analytic_Shift+1074.62


Analytic_Parameters = np.c_[Analytic_Shift, Analytic_Height, Analytic_Width]




# Showing the True, Network and Analytic Gaussians
Liner=np.linspace(1000,1105,1000)
GaussianNumber= 1000 # within [0,10000]
plt.plot(Liner, 
            montcarlpred[GaussianNumber,1] * np.exp(-(Liner-montcarlpred[GaussianNumber,0])*(Liner-montcarlpred[GaussianNumber,0])/(2*montcarlpred[GaussianNumber,2]*montcarlpred[GaussianNumber,2])),
            c='b', zorder=1)
plt.plot(Liner, 
            Analytic_Parameters[GaussianNumber,1] * np.exp(-(Liner-Analytic_Parameters[GaussianNumber,0])*(Liner-Analytic_Parameters[GaussianNumber,0])/(2*Analytic_Parameters[GaussianNumber,2]*Analytic_Parameters[GaussianNumber,2])),
            c='r', zorder=2)
plt.plot(Liner, 
            y_test[GaussianNumber,1] * np.exp(-(Liner-y_test[GaussianNumber,0])*(Liner-y_test[GaussianNumber,0])/(2*y_test[GaussianNumber,2]*y_test[GaussianNumber,2])), 
            c='green', zorder=0)
plt.scatter(X_Values, x_test[GaussianNumber], color='k')
plt.legend(['MC Predictions', 'Analytic Predictions', 'True Function'])
plt.show()


print(montcarlpred[GaussianNumber,0], Analytic_Parameters[GaussianNumber,0], y_test[GaussianNumber,0])
print(montcarlpred[GaussianNumber,1], Analytic_Parameters[GaussianNumber,1], y_test[GaussianNumber,1])
print(montcarlpred[GaussianNumber,2], Analytic_Parameters[GaussianNumber,2], y_test[GaussianNumber,2])



























