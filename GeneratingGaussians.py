import matplotlib.pyplot as plt
import numpy as np

# Initial parameters for testing purposes - standard normal
X_Values=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
Y_Values=np.zeros(np.size(X_Values), dtype='float32')
Shift=0.0
Peak_Height=1.0
Standard_Deviation=1.0
# calculating Y_Values
for j in range(np.size(X_Values)):
 Y_Values[j]=Peak_Height*np.exp(-(X_Values[j]-Shift)**2/(2*Standard_Deviation**2))

plot = plt.scatter(X_Values, Y_Values, s=20, edgecolors=("k"))
# This seems to work fine
'''
######################################################################################
'''
# Attempting to generate a training dataset
# First initialise X and preallocate Y to be filled once parameters are set
# Taking X values between 0 and 15 to begin with

X_Values = np.arange(0, 16, 1) # generates sequence of numbers between 0 and 16
Y_Values = np.zeros(np.size(X_Values), dtype='float32')
n = np.size(X_Values)

# Preallocating arrays to save the parameters for each runthrough
Training_Set_Size = 10000

# NO NEED TO PRE GENERATE ARRAYS ANYMORE
#Generated_Data = np.zeros(Training_Set_Size, dtype=object)
#Shift = np.zeros(Training_Set_Size, dtype='float32')
#Peak_Height = np.zeros(Training_Set_Size, dtype='float32')
#Standard_Deviation = np.zeros(Training_Set_Size, dtype='float32')

# Attempting to generate 10000 gaussian curves
#for i in range(Training_Set_Size):
    # Generating Random values for Shift, Height and Width, saving them in the
    # preallocated arrays Shift=[0 15], Height=[0 5], Deviation=[0 5]
Shift = np.random.uniform(low=0.0, high=15.0, size=Training_Set_Size) # adding size generates 10,000 values in one go
Peak_Height = np.random.uniform(low=0.0, high=5.0, size=Training_Set_Size)
Standard_Deviation = np.random.uniform(low=0.0, high=5.0, size=Training_Set_Size)
#    for j in range(n):
     # calculating the Y values of the randomly generated gaussian distributions
     # j refers to each element of the array thus only refers to the X and Y values,
     # i is referring to the above generated random parameters.

# Python uses vectorised operations, i.e. arrays can multiply arrays of numbers - no need for for loops
# Although, as X_Values has shape (16,) and e.g. Peak Height has shape (10000,) these cannot be multiplied
# A bit of broadcasting magic has to happen (https://numpy.org/doc/stable/user/theory.broadcasting.html#array-broadcasting-in-numpy)
Y_Values = Peak_Height*np.exp(-(X_Values[np.newaxis,].T-Shift)**2/(2*Standard_Deviation**2))+np.random.randn(len(X_Values),Training_set_size)*0.3
Generated_Data = Y_Values

# For some reason, it seems to be overwriting all previously generated data when the
# code is run, meaning that once the final gaussian is generated and the Y_values saved, 
# all of the data is equivalent to those final Y_values rather than maintaining the 
# 10000 different gaussians
Generated_Data

plt.scatter(X_Values, Y_Values, s=20, edgecolors=("k"))


# The true parameters for each gaussian, i.e. the target values for the later network. 
# Column 1 = Shift, Column 2 = Peak_Height, Column 3 = Standard_Deviation
True_Parameters=np.c_[Shift, Peak_Height, Standard_Deviation]

