#%% 
# Load module
import numpy as np 
import pandas as pd 
from matplotlib import pyplot as plt 
# %%
# Load data
ionosphere_col_names=['radar'+str(i) for i in range(34) ] + ['target']
ionosphere_data=pd.read_csv('ionosphere/ionosphere.data',names=ionosphere_col_names,header=None) # https://archive.ics.uci.edu/ml/datasets/ionosphere

adult_col_names=['age','workclass','fnlwgt','education','education-num','marital-status','occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country','target']
adult_data=pd.read_csv('adult_data_set/adult.data',names=adult_col_names,header=None) # https://archive.ics.uci.edu/ml/datasets/Adult

iris_col_names=['sepal_length','sepal_width','petal_length','petal_width','target']
iris_data=pd.read_csv('iris/iris.data',names=iris_col_names,header=None) # https://archive.ics.uci.edu/ml/datasets/Iris

car_col_names=['buying','maint','doors','persons','lug_boot','target']
car_data=pd.read_csv('car/car.data') # https://archive.ics.uci.edu/ml/datasets/Car+Evaluation
# %%
# Process ionosphere dataset
# Any missing values?
print('Missing values? ',np.any(ionosphere_data.isnull())) # No missing values
# Transform the target into binary value
ionosphere_data['target']=(ionosphere_data['target']=='g').astype(int)
# %%
# Plot the distribution of the data to check malformed features and discover the distribution
print('Data description\n',ionosphere_data.describe())
# We can see that 'radar1' are all zeros, so we remove it from the data
ionosphere_data=ionosphere_data.drop(columns="radar1")
# %%
