#%% 
# Load module
import numpy as np 
import pandas as pd 
from matplotlib import pyplot as plt 
import seaborn as sns
# %%
# Load data
ionosphere_col_names=['radar'+str(i) for i in range(34) ] + ['target']
ionosphere_data=pd.read_csv('ionosphere/ionosphere.data',names=ionosphere_col_names,header=None) # https://archive.ics.uci.edu/ml/datasets/ionosphere

adult_col_names=['age','workclass','fnlwgt','education','education-num','marital-status','occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country','target']
adult_data=pd.read_csv('adult_data_set/adult.data',names=adult_col_names,header=None) # https://archive.ics.uci.edu/ml/datasets/Adult

iris_col_names=['sepal_length','sepal_width','petal_length','petal_width','target']
iris_data=pd.read_csv('iris/iris.data',names=iris_col_names,header=None) # https://archive.ics.uci.edu/ml/datasets/Iris

car_col_names=['buying','maint','doors','persons','lug_boot','target']
car_data=pd.read_csv('car/car.data',names=car_col_names,header=None) # https://archive.ics.uci.edu/ml/datasets/Car+Evaluation
# %%
# Process ionosphere dataset
'''
All predictors are real values
'''
# Any missing values?
print('Missing values? ',np.any(ionosphere_data.isnull())) # No missing values
# Transform the target into binary value
ionosphere_data['target']=(ionosphere_data['target']=='g').astype(int)
# %%
# Give basic description to check malformed features and discover the distribution
print('Data description\n',ionosphere_data.describe())
# We can see that 'radar1' are all zeros, so we remove it from the data
ionosphere_data=ionosphere_data.drop(columns="radar1")
# %%
# Plot the distribution of the data and check malformed features and discover the distribution
# We plot the distribution of first three columns
ionosphere_first_three=ionosphere_data[['radar0','radar2','radar3']]
plt.plot(ionosphere_first_three)
plt.legend(['radar0','radar2','radar3'])
plt.show()
# It can be seen that the radars look like time serires
# %%
# What are the distribution of th positive and negative classes?

# How does the scatter plots of pair-wise features look-like for some subset of features
#iono_piar=sns.pairplot(ionosphere_data) # Time consuming!
#iono_piar.savefig("iono_piarplot.pdf")

# %%
# Process adult dataset
'''
Age: continuous count value
Workcalss: categorical
fnlwgt: continuous
education: categorical
education-num: continuous
marital-status: categorical
occupation: categorical
relationship: categorical
race: categorical
sex: binary
capital-gain: continuous.
capital-loss: continuous.
hours-per-week: continuous.
native-country: categorical 
'''
# Any missing values?
print('Missing values? ',np.any(adult_data.isnull())) # No missing values

# process categorical features
adult_wc={v:k for k,v in enumerate(adult_data['workclass'].unique())}
adult_edu={v:k for k,v in enumerate(adult_data['education'].unique())} 
adult_mar={v:k for k,v in enumerate(adult_data['marital-status'].unique())}
adult_ocp={v:k for k,v in enumerate(adult_data['occupation'].unique())}
adult_rel={v:k for k,v in enumerate(adult_data['relationship'].unique())}
adult_rac={v:k for k,v in enumerate(adult_data['race'].unique())}
adult_sex={v:k for k,v in enumerate(adult_data['sex'].unique())} 
adult_cnt={v:k for k,v in enumerate(adult_data['native-country'].unique())}
adult_cleanup_cols={
    'workclass': adult_wc,
    'education': adult_edu,
    'marital-status': adult_mar,
    'occupation': adult_ocp,
    'relationship': adult_rel,
    'race': adult_rac,
    'sex': adult_sex,
    'native-country': adult_cnt
}
adult_data.replace(adult_cleanup_cols,inplace=True)
adult_data['target']=(adult_data['target']==' >50K').astype(int)
# %%
# Process iris data
'''
sepal_length: real continuous
sepal_width: real continuous 
petal_length: real continuous
petal_width: real continuous
'''

# Any missing values?
print('Missing values? ',np.any(iris_data.isnull())) # No missing values 

iris_target={v:k for k,v in enumerate(iris_data['target'].unique())}
iris_cleanup_cols={
    'target': iris_target
}
iris_data.replace(iris_cleanup_cols,inplace=True)

# %%
# Process car data
'''
buying: categorical
maint: categorical
doors: categorical
persons: categorical
lug_boot: categorical
safety: categorical
'''
# Any missing values?
print('Missing values? ',np.any(car_data.isnull())) # No missing values 

car_buy={v:k for k,v in enumerate(car_data['buying'].unique())}
car_mnt={v:k for k,v in enumerate(car_data['maint'].unique())}
car_dor={v:k for k,v in enumerate(car_data['doors'].unique())}
car_per={v:k for k,v in enumerate(car_data['persons'].unique())}
car_lug={v:k for k,v in enumerate(car_data['lug_boot'].unique())}
car_target={'unacc':0,'acc':1,'good':2,'vgood':3}
car_cleanup_cols={
    'buying':car_buy,
    'maint': car_mnt,
    'doors': car_dor,
    'persons': car_per,
    'lug_boot': car_lug,
    'target': car_target
}
car_data.replace(car_cleanup_cols,inplace=True)

# %%

# Save data
np.save('ionosphere_data_cleaned.npy',ionosphere_data)
np.save('adult_data_cleaned.npy',adult_data)
np.save('iris_data_cleaned.npy',iris_data)
np.save('car_data_cleaned.npy',car_data)

# %%
