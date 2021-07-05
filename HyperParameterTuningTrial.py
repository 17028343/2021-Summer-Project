import numpy as np
import tensorflow as tf
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import GridSearchCV

from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding, Flatten, LeakyReLU, BatchNormalization, Dropout
from keras.activations import relu, sigmoid


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


# I changed the model definition slightly, you were using x_train inside the function,
# but not passing it in. This can have odd behaviour. So I have explicitly specified the
# value instead. RM
def create_model(layers, activation, input_shape=16):
    model = Sequential()
    for i, nodes in enumerate(layers):
        if i==0:
            model.add(Dense(nodes,input_dim=input_shape))
            model.add(Activation(activation))
            model.add(Dropout(0.05))
        else:
            model.add(Dense(nodes))
            model.add(Activation(activation))
            model.add(Dropout(0.05))
            
    model.add(Dense(units = 3, kernel_initializer= 'glorot_uniform', activation = 'linear'))
    
    model.compile(optimizer='adam', loss='mean_squared_error',metrics=['accuracy'])
    return model

# changed this from classifier to regressor
model = KerasRegressor(build_fn=create_model, verbose=0)

layers = [[50], [64, 10]]
activations = ['sigmoid', 'relu']
# removed parameters from here, as you are not searching over epochs and batch size
# It could be that this was causing the problem, with grid search function expecting to find
# corresponding arguments in the model defintion. RM
param_grid = dict(layers=layers, activation=activations)


grid = GridSearchCV(estimator=model, param_grid=param_grid,cv=5)

earlystop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)

# added epochs here RM
grid_result = grid.fit(x_train, y_train, epochs=100, callbacks=earlystop)

# Getting this error after running the code for 10-ish minutes. Not sure how to fix
# Cannot clone object <tensorflow.python.keras.wrappers.scikit_learn.KerasClassifier object at 0x0000016D454847F0>, as the constructor either does not set or modifies parameter layers

[grid_result.best_score_,grid_result.best_params_]
