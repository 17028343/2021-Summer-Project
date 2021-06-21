import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import regularizers
import scipy as sp
from scipy.optimize import curve_fit
import seaborn as sns


# The X-vals of each data point for the gaussian distributions
X_Values = np.arange(0, 16, 1)

# allocating the size of the training data set
Training_Set_Size = 100000

# Randomly Setting Parameters to generate the different gaussians
np.random.seed(3)
Shift = np.random.uniform(low=0.0, high=15.0, size=Training_Set_Size) 
np.random.seed(2)
Peak_Height = np.random.uniform(low=0.0, high=5.0, size=Training_Set_Size)
np.random.seed(1)
Standard_Deviation = np.random.uniform(low=0.0, high=5.0, size=Training_Set_Size)

# Calculating the y-values for each gaussian in the training data set
np.random.seed(0)
Y_Values = Peak_Height*np.exp(-(X_Values[np.newaxis,].T-Shift)**2/(2*Standard_Deviation**2))+np.random.randn(len(X_Values),Training_Set_Size)*0.3
Generated_Data = Y_Values

# The true values of each distribution's shift, height and width
True_Parameters=np.c_[Shift, Peak_Height, Standard_Deviation]

# Building the Network
earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
GaussianNet = tf.keras.models.Sequential([
  tf.keras.layers.Dense(16, activation='sigmoid', kernel_regularizer=regularizers.l2(0.0000001)),
  tf.keras.layers.Dropout(0.1, seed=8),
  tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99),
  tf.keras.layers.Dense(64, activation='sigmoid', kernel_regularizer=regularizers.l2(0.0000001)),
  tf.keras.layers.Dropout(0.1, seed=8),
  tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99),
  tf.keras.layers.Dense(10, activation='sigmoid', kernel_regularizer=regularizers.l2(0.0000001)),
  tf.keras.layers.Dropout(0.1, seed=8),
  tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99),
  tf.keras.layers.Dense(3, activation='linear')
])

GaussianNet.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

# Training the Network
ModelData=GaussianNet.fit(np.transpose(Generated_Data), True_Parameters, validation_split=0.1, epochs=500, callbacks=earlystop)



# allocating the size of the validation data set
Validation_Set_Size = 1000

# Randomly Setting Parameters to generate the validation gaussians
np.random.seed(6)
Val_Shift = np.random.uniform(low=0.0, high=15.0, size=Validation_Set_Size) 
np.random.seed(7)
Val_Peak_Height = np.random.uniform(low=0.0, high=5.0, size=Validation_Set_Size)
np.random.seed(5)
Val_Standard_Deviation = np.random.uniform(low=0.0, high=5.0, size=Validation_Set_Size)

# Calculating the y-values for each validation gaussian 
np.random.seed(4)
Validation_Y_Values = Val_Peak_Height*np.exp(-(X_Values[np.newaxis,].T-Val_Shift)**2/(2*Val_Standard_Deviation**2))+np.random.randn(len(X_Values),Validation_Set_Size)*0.3

# The true values of each distribution's shift, height and width
Validation_Parameters = np.c_[Val_Shift, Val_Peak_Height, Val_Standard_Deviation]

# Getting the predictions made by the network for the validation set
NETWORKPredictions = GaussianNet.predict(np.transpose(Validation_Y_Values))


Net_Errors=NETWORKPredictions-Validation_Parameters

NetPercentage_Errors=100*(Net_Errors)/Validation_Parameters
# Determining Accuracy of Network Predictions
Within30Net=100*np.size(NetPercentage_Errors[NetPercentage_Errors<30])/np.size(NetPercentage_Errors)
Within10Net=100*np.size(NetPercentage_Errors[NetPercentage_Errors<10])/np.size(NetPercentage_Errors)
Within1Net=100*np.size(NetPercentage_Errors[NetPercentage_Errors<1])/np.size(NetPercentage_Errors)



# Traditional Method
# Define the true function that we are trying to fit to
def func(x, height, shift, standev) :
    return height * np.exp(-(x-shift)*(x-shift)/(2*standev*standev))


# Preallocate the holder of the predictions
TraditionalPredictions=[[0]*3 for i in range(Validation_Set_Size)]
# Initiate a counter to see if there's a problem
Counter=0
for i in np.arange(0,Validation_Set_Size,1):
    popt, pcov = curve_fit(func, X_Values, Validation_Y_Values[:,i], p0=(2.5, 7.5, 2.5), bounds=(0, [5, 15, 5]))
    TraditionalPredictions[i]=popt
    Counter+=1

TraditionalPredictions=np.array(TraditionalPredictions,dtype="float64")
TraditionalPredictions=TraditionalPredictions[:, [1, 0, 2]]
TraditionalShift=TraditionalPredictions[:,0]
TraditionalHeight=TraditionalPredictions[:,1]
TraditionalWidth=TraditionalPredictions[:,2]




TradErrors=TraditionalPredictions-Validation_Parameters
TraditionalPercentage_Errors=100*(TradErrors)/Validation_Parameters
# Determining Accuracy of Traditional Method
Within30Trad=100*np.size(TraditionalPercentage_Errors[TraditionalPercentage_Errors<30])/np.size(TraditionalPercentage_Errors)
Within10Trad=100*np.size(TraditionalPercentage_Errors[TraditionalPercentage_Errors<10])/np.size(TraditionalPercentage_Errors)
Within1Trad=100*np.size(TraditionalPercentage_Errors[TraditionalPercentage_Errors<1])/np.size(TraditionalPercentage_Errors)


NetShiftErrs=Net_Errors[:,0]
NetHeightErrs=Net_Errors[:,1]
NetStanDevErrs=Net_Errors[:,2]

TradShiftErrs=TradErrors[:,0]
TradHeightErrs=TradErrors[:,1]
TradStanDevErrs=TradErrors[:,2]

plt.scatter(NETWORKPredictions[:,0], Val_Shift, c='c', edgecolors="k", linewidths=0.2)
plt.scatter(TraditionalShift, Val_Shift, c='b', edgecolors="k", linewidths=0.2)
plt.xlim(0,25)
plt.xlabel('Predicted Shift Values')
plt.ylabel('True Shift Values')
plt.legend(['Network Predictions', 'Traditional Predictions'], loc='right')
plt.show()

plt.scatter(NETWORKPredictions[:,1], Val_Peak_Height, c='r', edgecolors="k", linewidths=0.2)
plt.scatter(TraditionalHeight, Val_Peak_Height, c='maroon', edgecolors="k", linewidths=0.2)
plt.xlim(0,10)
plt.xlabel('Predicted Height Values')
plt.ylabel('True Height Values')
plt.legend(['Network Predictions', 'Traditional Predictions'], loc='right')
plt.plot([0,5], [0,5])
plt.show()

plt.scatter(NETWORKPredictions[:,2], Val_Standard_Deviation, c='lime', edgecolors="k", linewidths=0.2)
plt.scatter(TraditionalWidth, Val_Standard_Deviation, c='green', edgecolors="k", linewidths=0.2)
plt.xlim(0,10)
plt.xlabel('Predicted SD Values')
plt.ylabel('True SD Values')
plt.legend(['Network Predictions', 'Traditional Predictions'], loc='right')
plt.show()

NetShiftErrAve=np.mean(NetShiftErrs)
TradShiftErrAve=np.mean(TradShiftErrs)

NetHeightErrAve=np.mean(NetHeightErrs)
TradHeightErrAve=np.mean(TradHeightErrs)

NetStanDevErrAve=np.mean(NetStanDevErrs)
TradStanDevErrAve=np.mean(TradStanDevErrs)

sns.kdeplot(NetShiftErrs, color="c")
sns.kdeplot(TradShiftErrs, color="b")
plt.xlabel('Error in Shift Prediction')
plt.ylim(0,1.0)
plt.legend(['Network Errors', 'Traditional Errors'], loc='upper right')
plt.annotate('Traditional Method Mean Error', (-15, 0.95))
plt.annotate(f'{TradShiftErrAve}', (-13, 0.88))
plt.annotate('Network Mean Error', (-12.8, 0.78))
plt.annotate(f'{NetShiftErrAve}', (-14, 0.71))
plt.show()

sns.kdeplot(NetHeightErrs, color="r")
sns.kdeplot(TradHeightErrs, color="maroon")
plt.xlabel('Error in Height Prediction')
plt.legend(['Network Errors', 'Traditional Errors'], loc='upper right')
plt.annotate('Traditional Method Mean Error', (-5.5, 0.95))
plt.annotate(f'{TradHeightErrAve}', (-5, 0.88))
plt.annotate('Network Mean Error', (-4.5, 0.78))
plt.annotate(f'{NetHeightErrAve}', (-5, 0.71))
plt.show()

sns.kdeplot(NetStanDevErrs, color="lime")
sns.kdeplot(TradStanDevErrs, color="green")
plt.xlabel('Error in StanDev Prediction')
plt.legend(['Network Errors', 'Traditional Errors'], loc='upper right')
plt.annotate('Traditional Method Mean Error', (-5.7, 0.95))
plt.annotate(f'{TradStanDevErrAve}', (-5.5, 0.88))
plt.annotate('Network Mean Error', (-5.0, 0.78))
plt.annotate(f'{NetStanDevErrAve}', (-5.5, 0.71))
plt.show()


Liner=np.linspace(0,15,1000)
GaussianNumber=109
plt.scatter(X_Values, Validation_Y_Values[:,GaussianNumber], marker='o', c='lime', ec='k', lw=0.5, zorder=3)
plt.plot(Liner, 
            NETWORKPredictions[GaussianNumber,1] * np.exp(-(Liner-NETWORKPredictions[GaussianNumber,0])*(Liner-NETWORKPredictions[GaussianNumber,0])/(2*NETWORKPredictions[GaussianNumber,2]*NETWORKPredictions[GaussianNumber,2])),
            c='b', zorder=1)
plt.plot(Liner, 
            TraditionalHeight[GaussianNumber] * np.exp(-(Liner-TraditionalShift[GaussianNumber])*(Liner-TraditionalShift[GaussianNumber])/(2*TraditionalWidth[GaussianNumber]*TraditionalWidth[GaussianNumber])),
            c='r', zorder=2)
plt.plot(Liner, 
            Val_Peak_Height[GaussianNumber] * np.exp(-(Liner-Val_Shift[GaussianNumber])*(Liner-Val_Shift[GaussianNumber])/(2*Val_Standard_Deviation[GaussianNumber]*Val_Standard_Deviation[GaussianNumber])), 
            c='green', zorder=0)
plt.legend(['True function', 'Network Predictions', 'Traditional Predictions', 'True Values'])
plt.show()


f"True Parameters: {Validation_Parameters[GaussianNumber, :]}, Network Parameters: {NETWORKPredictions[GaussianNumber, :]}, Traditional Parameters: {TraditionalPredictions[GaussianNumber, :]}"

