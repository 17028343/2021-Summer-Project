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


# Creating the dense network
GaussianNet = tf.keras.models.Sequential([
  tf.keras.layers.Dense(16, activation='sigmoid'),
  tf.keras.layers.Dense(528, activation='sigmoid'),
  tf.keras.layers.Dense(256, activation='sigmoid'),
  tf.keras.layers.Dense(128, activation='sigmoid'),
  tf.keras.layers.Dense(64, activation='sigmoid'),
  tf.keras.layers.Dense(10, activation='sigmoid'),
  tf.keras.layers.Dense(3, activation='linear')
])

GaussianNet.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

ModelData=GaussianNet.fit(np.transpose(Generated_Data), True_Parameters, validation_split=0.1, epochs=30)


# Plotting the Accuracy of the Model over time, both training and validation data
plt.plot(ModelData.history['accuracy'])
plt.plot(ModelData.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs after 30')
plt.legend(['Training Set', 'Validation Set'], loc='lower right')
plt.grid()
plt.show()

# Plotting the Loss of the Model over time, both training and validation data
plt.plot(ModelData.history['loss'])
plt.plot(ModelData.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs after 30')
plt.legend(['Training Set', 'Validation Set'], loc='upper right')
plt.grid()
plt.show()
































