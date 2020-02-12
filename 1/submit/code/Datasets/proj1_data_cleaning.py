#%% 
# Load module
import numpy as np 
import pandas as pd 
from matplotlib import pyplot as plt 
from sklearn.decomposition import PCA
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
plt.style.use('seaborn')
# %%
# Load data
ionosphere_col_names=['radar'+str(i) for i in range(34) ] + ['target']
ionosphere_data=pd.read_csv('ionosphere/ionosphere.data',names=ionosphere_col_names,header=None) # https://archive.ics.uci.edu/ml/datasets/ionosphere

adult_col_names=['age','workclass','fnlwgt','education','education-num','marital-status','occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country','target']
adult_data=pd.read_csv('adult_data_set/adult.data',names=adult_col_names,header=None) # https://archive.ics.uci.edu/ml/datasets/Adult

iris_col_names=['sepal_length','sepal_width','petal_length','petal_width','target']
iris_data=pd.read_csv('iris/iris.data',names=iris_col_names,header=None) # https://archive.ics.uci.edu/ml/datasets/Iris

car_col_names=['buying','maint','doors','persons','lug_boot','safety','target']
car_data=pd.read_csv('car/car.data',names=car_col_names,header=None) # https://archive.ics.uci.edu/ml/datasets/Car+Evaluation

# %% Function definition to clean continuous features

def clean_continuous(input_feature_array,pca_flag=False, pca_dim=10,disc_bins=11):
    input_feature_array_red = input_feature_array.copy()

    if pca_flag:
        pca=PCA(pca_dim) # Reduce to 10 dimensions, we can hyper-parameter search later on
        input_feature_array_red = pca.fit_transform(input_feature_array)

    input_feature_array_red_norm = MinMaxScaler().fit_transform(input_feature_array_red)
    bins=np.linspace(0,1,disc_bins)# The real values are normalized in to [0,1]
    input_feature_array_red_norm_disc = input_feature_array_red_norm.copy()
    # Discretize the real value
    for i in range(input_feature_array_red_norm.shape[1]):
        input_feature_array_red_norm_disc[:,i]=np.digitize(input_feature_array_red_norm[:,i],bins=bins)
    return input_feature_array_red_norm_disc
# %% Process ionosphere dataset
'''
All predictors are real values
'''
# Any missing values?
print('Missing values? ',np.any(ionosphere_data.isnull())) # No missing values
# Transform the target into binary value
ionosphere_data['target']=(ionosphere_data['target']=='g').astype(int)
# %% Plot the distribution of good and bad class
iono_g = ionosphere_data[list(ionosphere_data.columns)[:-1]].loc[ionosphere_data['target']==1]
iono_b = ionosphere_data[list(ionosphere_data.columns)[:-1]].loc[ionosphere_data['target']==0]
fig,ax = plt.subplots(figsize=(8,6))
ax.plot(iono_g.mean(0),label="Good class")
ax.plot(iono_b.mean(0),label="Bad class")
iono_g_up=iono_g.mean(0)[iono_g.mean(0)>=iono_g.mean(0).mean()]
iono_g_dn=iono_g.mean(0)[iono_g.mean(0)<iono_g.mean(0).mean()]
iono_g_up_p = np.zeros(2*len(iono_g_up)-1)
iono_g_dn_p = np.zeros(2*len(iono_g_dn)-1)

for i in range(0,len(iono_g_up_p),1):
    if i % 2 == 1:
        iono_g_up_p[i]=np.mean([iono_g_up[int((i-1)/2)],iono_g_up[int((i+1)/2)]])
        iono_g_dn_p[i]=np.mean([iono_g_dn[int((i-1)/2)],iono_g_dn[int((i+1)/2)]])
    else:
        iono_g_up_p[i]=iono_g_up[int(i/2)]
        iono_g_dn_p[i]=iono_g_dn[int(i/2)]

x=np.arange(0,len(iono_g.mean(0)),1)
ax.fill_between(x,np.append(iono_g_up_p,0.4), np.append(0,iono_g_dn_p), color='gray', alpha=0.2)
ax.set_xlabel('Radar No.')
ax.set_ylabel('Average antennas power')
ax.set_title('Average feature distribution of good & bad class in Ionoshpere data ')
ax.set_xticklabels([str(i) for i in np.arange(0,34,1)])
ax.legend()
plt.tight_layout()
plt.savefig('iono_feat_dist.pdf')
# %%
# Give basic description to check malformed features and discover the distribution
print('Data description\n',ionosphere_data.describe())
# We can see that 'radar1' are all zeros, so we remove it from the data
ionosphere_data=ionosphere_data.drop(columns="radar1")

# %% Dimension reduction, min-max normalization and Discretization
ionosphere_feat_red_norm_disc = clean_continuous(ionosphere_data[list(ionosphere_data.columns)[:-1]],pca_flag=True)


# %% Plot the distribution of the data and check malformed features and discover the distribution
# We plot the distribution of first three columns
ionosphere_first_three=ionosphere_data[['radar0','radar2','radar3']]
plt.plot(ionosphere_first_three)
plt.legend(['radar0','radar2','radar3'])
plt.show()
# It can be seen that the radars look like time serires
# %%
# What are the distribution of th positive and negative classes?

# How does the scatter plots of pair-wise features look-like for some subset of features
#iono_pair=sns.pairplot(ionosphere_data[list(ionosphere_data.columns[:-1])]).set(title='Ionosphere dataset pairplot')
#iono_pair.savefig("iono_pairplot.pdf")

# %%
# Process adult dataset
'''
age: continuous count value
workcalss: categorical
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
# Process continuous value
adult_continuous_cols = ['age','fnlwgt','education-num','capital-gain','capital-loss','hours-per-week']
adult_feat_red_norm_disc = clean_continuous(adult_data[adult_continuous_cols])
adult_data[adult_continuous_cols]=adult_feat_red_norm_disc
adult_data[adult_continuous_cols]=adult_data[adult_continuous_cols].astype('int64')

# %% Process iris data
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

iris_continuous_cols = list(iris_data.columns)[:-1]
iris_feat_red_norm_disc = clean_continuous(iris_data[iris_continuous_cols])
iris_data[iris_continuous_cols]=iris_feat_red_norm_disc
iris_data[iris_continuous_cols]=iris_data[iris_continuous_cols].astype('int64')

# %% Pairplot
iris_pair=sns.pairplot(iris_data[list(iris_data.columns[:-1])]).set(title='Iris dataset pairplot')
iris_pair.savefig("iris_pairplot.pdf")
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
car_saf={v:k for k,v in enumerate(car_data['safety'].unique())}
car_target={'unacc':0,'acc':1,'good':2,'vgood':3}
car_cleanup_cols={
    'buying':car_buy,
    'maint': car_mnt,
    'doors': car_dor,
    'persons': car_per,
    'lug_boot': car_lug,
    'safety': car_saf,
    'target': car_target
}
car_data.replace(car_cleanup_cols,inplace=True)
# %% Pairplot
car_pair=sns.pairplot(car_data[list(car_data.columns[:-1])]).set(title='Car dataset pairplot')
car_pair.savefig("car_pairplot.pdf")
# %% Save data
ionosphere_feat_norm_disc = clean_continuous(ionosphere_data[list(ionosphere_data.columns)[:-1]],pca_flag=False)

ionosphere_data_cleaned = pd.DataFrame(np.column_stack((ionosphere_feat_norm_disc,ionosphere_data["target"])), columns=list(np.delete(  np.array(ionosphere_col_names),1)) )

ionosphere_data_pca=pd.DataFrame(np.column_stack((ionosphere_feat_red_norm_disc,ionosphere_data["target"])),columns=['reduced_'+str(i) for i in range(10)]+['target'])

np.save('ionosphere_cleaned.npy', ionosphere_data_cleaned )
np.save('ionosphere_pca.npy',ionosphere_data_pca)
np.save('ionosphere_features.npy',ionosphere_feat_red_norm_disc)
np.save('ionosphere_target.npy',ionosphere_data["target"])

np.save('adult_data_cleaned.npy',adult_data)
np.save('adult_data_features.npy',adult_data[list(adult_data.columns)[:-1]])
np.save('adult_data_target.npy',adult_data[list(adult_data.columns)[-1]])

np.save('iris_data_cleaned.npy',iris_data)
np.save('iris_data_features.npy',iris_data[list(iris_data.columns)[:-1]])
np.save('iris_data_target.npy',iris_data[list(iris_data.columns)[-1]])

np.save('car_data_cleaned.npy',car_data)
np.save('car_data_features.npy',car_data[list(car_data.columns)[:-1]])
np.save('car_data_target.npy',car_data[list(car_data.columns)[-1]])
