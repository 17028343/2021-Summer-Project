import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import utils
from tensorflow.keras import regularizers

# The X-vals of each data point for the gaussian distributions
X_Values = np.arange(0, 16, 1)

# allocating the size of the training data set
Training_Set_Size = 100000

# Randomly Setting Parameters to generate the different gaussians
Shift = np.random.uniform(low=0.0, high=15.0, size=Training_Set_Size) 
Peak_Height = np.random.uniform(low=0.0, high=5.0, size=Training_Set_Size)
Standard_Deviation = np.random.uniform(low=0.0, high=5.0, size=Training_Set_Size)

# Calculating the y-values for each gaussian in the training data set
Y_Values = Peak_Height*np.exp(-(X_Values[np.newaxis,].T-Shift)**2/(2*Standard_Deviation**2))+np.random.randn(len(X_Values),Training_Set_Size)*0.3
Generated_Data = Y_Values

# The true values of each distribution's shift, height and width
True_Parameters=np.c_[Shift, Peak_Height, Standard_Deviation]


# Creating the dense network - Early Stopping and Batch Normalization
earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
GaussianNet = tf.keras.models.Sequential([
  tf.keras.layers.Dense(16, activation='sigmoid', kernel_regularizer=regularizers.l2(0.0000001)),
  tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99),
  tf.keras.layers.Dense(64, activation='sigmoid', kernel_regularizer=regularizers.l2(0.0000001)),
  tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99),
  tf.keras.layers.Dense(10, activation='sigmoid', kernel_regularizer=regularizers.l2(0.0000001)),
  tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99),
  tf.keras.layers.Dense(3, activation='linear', kernel_regularizer=regularizers.l2(0.0000001))
])

GaussianNet.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

ModelData=GaussianNet.fit(np.transpose(Generated_Data), True_Parameters, validation_split=0.1, epochs=500, callbacks=earlystop)

# GaussianNet.layers[2].weights


# Plotting the Accuracy of the Model over time, both training and validation data
plt.plot(ModelData.history['accuracy'])
plt.plot(ModelData.history['val_accuracy'])
plt.title('Model Accuracy - L2 1e-7 - 16, 64, 10')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['Training Set', 'Validation Set'], loc='lower right')
plt.grid()
plt.show()

# Plotting the Loss of the Model over time, both training and validation data
plt.plot(ModelData.history['loss'])
plt.plot(ModelData.history['val_loss'])
plt.title('Model Loss - L2 1e-7 - 16, 64, 10')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['Training Set', 'Validation Set'], loc='upper right')
plt.grid()
plt.show()



'''
MAKING PREDICTIONS FOR A BETTER WAY OF ASSESSING THE NETWORK'S SUCCESS RATE
'''

# allocating the size of the validation data set
Validation_Set_Size = 1000

# Randomly Setting Parameters to generate the validation gaussians
Val_Shift = np.random.uniform(low=0.0, high=15.0, size=Validation_Set_Size) 
Val_Peak_Height = np.random.uniform(low=0.0, high=5.0, size=Validation_Set_Size)
Val_Standard_Deviation = np.random.uniform(low=0.0, high=5.0, size=Validation_Set_Size)

# Calculating the y-values for each validation gaussian 
Validation_Y_Values = Val_Peak_Height*np.exp(-(X_Values[np.newaxis,].T-Val_Shift)**2/(2*Val_Standard_Deviation**2))+np.random.randn(len(X_Values),Validation_Set_Size)*0.3
Validation_Data = Validation_Y_Values

# The true values of each distribution's shift, height and width
Validation_Parameters = np.c_[Val_Shift, Val_Peak_Height, Val_Standard_Deviation]

# Getting the predictions made by the network for the validation set
Predictions = GaussianNet.predict(np.transpose(Validation_Data))



# PLOTTING THE TRUE PARAMETERS AGAINST NETWORK PREDICTIONS

# Plotting the shift predictions against the true shift values
plt.scatter(Predictions[:,0], Validation_Parameters[:,0], s=20, edgecolors=("k"), c=("b"))
plt.ylabel('True Shift')
plt.xlabel('Predicted Shift')
plt.axis('square')
plt.plot([-1.0, 16.0], [-1.0, 16.0], c=("c"), linewidth=2)
plt.show()

# Plotting the height predictions against the true height parameters
plt.scatter(Predictions[:,1], Validation_Parameters[:,1], s=20, edgecolors=("k"), c=("firebrick"))
plt.ylabel('True Height')
plt.xlabel('Predicted Height')
plt.axis('square')
plt.plot([-1.0, 6.0], [-1.0, 6.0], c=("r"), linewidth=2)
plt.show()

# Plotting the standard deviation predictions against the true standard deviation parameters
plt.scatter(Predictions[:,2], Validation_Parameters[:,2], s=20, edgecolors=("k"), c=("darkgreen"))
plt.ylabel('True Standard Deviation')
plt.xlabel('Predicted Standard Deviation')
plt.axis('square')
plt.plot([-1.0, 6.0], [-1.0, 6.0], c=("lime"), linewidth=2)
plt.show()





# Calculating the actual differences
Errors=(Predictions-Validation_Parameters)
Err_Shift=Errors[:,0]
Err_Height=Errors[:,1]
Err_Standard_Deviation=Errors[:,2]

# PLOTTING THE TRUE ERRORS OF THE PREDICTIONS

# Plotting the actual error of the shift predictions
plt.scatter(np.arange(0,Validation_Set_Size,1), Err_Shift, s=20, edgecolors=("k"), c=("c"))
plt.xlim(0.0, Validation_Set_Size)
plt.ylabel('Error in Shift Prediction')
plt.xlabel('Gaussian Model')
plt.show()

# Plotting the actual error of the height predictions
plt.scatter(np.arange(0,Validation_Set_Size,1), Err_Height, s=20, edgecolors=("k"), c=("r"))
plt.xlim(0.0, Validation_Set_Size)
plt.ylabel('Error in Height Prediction')
plt.xlabel('Gaussian Model')
plt.show()

# Plotting the actual error of the Standard Deviation predictions
plt.scatter(np.arange(0,Validation_Set_Size,1), Err_Standard_Deviation, s=20, edgecolors=("k"), c=("lime"))
plt.xlim(0.0, Validation_Set_Size)
plt.ylabel('Error in SD Prediction')
plt.xlabel('Gaussian Model')
plt.show()


# Worth observing that the actual differences seem to be normally distributed?
# Shift Errors normal parameters
print('Shift Error Parameters', (np.mean(Err_Shift),np.std(Err_Shift)))
plt.hist(Err_Shift, color=("c"), ec=("k"))
plt.title('Histogram of Shift Errors')
plt.show()

# Shift Errors normal parameters
print((np.mean(Err_Height),np.std(Err_Height)))
plt.hist(Err_Height, color=("r"), ec=("k"))
plt.title('Histogram of Height Errors')
plt.show()

# Shift Errors normal parameters
print((np.mean(Err_Standard_Deviation),np.std(Err_Standard_Deviation)))
plt.hist(Err_Standard_Deviation, color=("lime"), ec=("k"))
plt.title('Histogram of Standard Deviation Errors')
plt.show()

# All Three Histograms
plt.hist(Err_Shift, color=("c"), ec=("k"), zorder=1)
plt.hist(Err_Standard_Deviation, color=("lime"), ec=("k"), zorder=2, alpha=0.8)
plt.hist(Err_Height, color=("r"), ec=("k"), zorder=3, alpha=0.5)
plt.tick_params(axis='y', which='both', left=False, right=False, labelleft=False) 
plt.title('Histograms of all Parameter Errors')
plt.legend(['Shift', 'Standard Deviation', 'Height'], loc='upper right')
plt.show()



# Combination of Scatter and Hist for the Shift Errors
plt.hist(Err_Shift, color=("b"), ec=("k"), orientation="horizontal", zorder=1)
plt.scatter(np.arange(0,Validation_Set_Size,1), Err_Shift, s=20, edgecolors=("k"), c=("c"), zorder=2, lw=0.5)
plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False) 
plt.ylabel('Error in Shift Predictions')
plt.title('The Error in Network Shift Predictions, with Histogram to show density')
plt.show()

# Combination of Scatter and Hist for the Height Errors
plt.hist(Err_Height, color=("firebrick"), ec=("k"), orientation="horizontal", zorder=1)
plt.scatter(np.arange(0,Validation_Set_Size,1), Err_Height, s=20, edgecolors=("k"), c=("r"), alpha=0.7, zorder=2, lw=0.5)
plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False) 
plt.ylabel('Error in Height Predictions')
plt.title('The Error in Network Height Predictions, with Histogram to show density')
plt.show()

# Combination of Scatter and Hist for the Errors
plt.hist(Err_Standard_Deviation, color=("darkgreen"), ec=("k"), orientation="horizontal", zorder=1)
plt.scatter(np.arange(0,Validation_Set_Size,1), Err_Standard_Deviation, s=20, edgecolors=("k"), c=("lime"), alpha=0.7, zorder=2, lw=0.5)
plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False) 
plt.ylabel('Error in SD Predictions')
plt.title('The Error in Network SD Predictions, with Histogram to show density')
plt.show()














