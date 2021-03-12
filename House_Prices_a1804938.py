#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.pyplot import xticks

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
pd.pandas.set_option('display.max_columns', None)

from sklearn import metrics
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor


# Data Analysis

# In[3]:


# load the dataset
#house = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")
#house.head()
df = pd.read_csv("/Users/user/Downloads/all/train.csv")
df.head()


# In[4]:


df.shape


# We can see that there are 81 features and this dataset might require lot of data analysis.

# In[5]:


# check the dataset
df.info()


# From the above output we can see that there are lots of null values present in the dataset.
# These will be checked and handled during data analysis.

# In[13]:


missing_colmns = [(c, df[c].isna().mean()*100) for c in df]
missing_colmns = pd.DataFrame(missing_colmns, columns=["column_name", "percentage"])
missing_colmns = missing_colmns[missing_colmns.percentage > 0]
display(missing_colmns.sort_values("percentage", ascending=False))


# Below is the output variable - "SalePrice"

# In[6]:


# target variable "SalePrice"
df["SalePrice"].describe()


# We can notice that there are no negative values for sale price in the dataset.

# In[7]:


#distribution of saleprice
sns.distplot(df.SalePrice)


# Saleprice seems to be skewed, This need to be handled else this will adversly impact our model

# In[8]:


#correlation matrix
corrmat = df.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);


# In[12]:


#saleprice correlation matrix
k = 10 #number of variables for heatmap
cols_corr = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(df[cols_corr].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols_corr.values, xticklabels=cols_corr.values)
plt.show()


# In[11]:


#scatterplot
sns.set()
cols_scatter = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(df[cols_scatter], height = 2.5)
plt.show();


# In[7]:


# lets drop Id because its of no use to us
df.drop("Id",1,inplace = True)


# In[8]:


# Let's display the variables with more than 0 null values
null_cols = []
for col in df.columns:
    if df[col].isnull().sum() > 0 :
        print("Column",col, "has", df[col].isnull().sum(),"null values")    
        null_cols.append(col)


# In[9]:


# lets visualize the null vaues
plt.figure(figsize=(12,10))
sns.barplot(x=df[null_cols].isnull().sum().index, y=df[null_cols].isnull().sum().values)
xticks(rotation=45)
plt.show()


# After checking the data dictionary, these are not actually the null values, rather these are the features which are not present in the house.
# 
# For example, let check the field Alley, Value "NA: here means house has "No Alley Access"

# In[10]:


# lets check if these null values actually have any relation with the target variable

df_eda = df.copy()

for col in null_cols:
    df_eda[col] = np.where(df_eda[col].isnull(), 1, 0)  

# lets see if these null values have to do anything with the sales price
plt.figure(figsize = (16,48))
for idx,col in enumerate(null_cols):
    plt.subplot(10,2,idx+1)
    sns.barplot(x = df_eda.groupby(col)["SalePrice"].median(),y =df_eda["SalePrice"])
plt.show()


# From the above graphs, we can clearly see that the null values have strong relation with the SalePrice, hence we can niether drop the columns with null values, nor we can drop the rows with null values.

# # Null Values Treatment

# In[11]:


# all missing values for the categorical columns will be replaced by "None"
# all missing values for the numeric columns will be replaced by median of that field

for col in df.columns:
    if df[col].dtypes == 'O':
        df[col] = df[col].replace(np.nan,"None")
    else:
        df[col] = df[col].replace(np.nan,df[col].median())


# In[12]:


# making list of date variables
yr_vars = []
for col in df.columns:
    if "Yr" in col or "Year" in col:
        yr_vars.append(col)

yr_vars = set(yr_vars)
yr_vars


# Let's check relation of these fields with the target variable

# In[13]:


plt.figure(figsize = (15,12))
for idx,col in enumerate(yr_vars):
    plt.subplot(2,2,idx+1)
    plt.plot(df.groupby(col)["SalePrice"].median())
    plt.xlabel(col)
    plt.ylabel("SalePrice")


# Make a note of the trend of sale price with the field "YrSold", it shows a decreasing trend which seems unreal in real state scenario, price is expected to increase as the time passes by, but here it shows opposite. Let's handle this by creating "Age" variables from these variables

# In[14]:


# creating age variables
df['HouseAge'] =  df['YrSold'] - df['YearBuilt']
# age of master after remodelling
df['RemodAddAge'] = df['YrSold'] - df['YearRemodAdd']
# creating age of the garage from year built of the garage to the sale of the master
df['GarageAge'] = df['YrSold'] - df['GarageYrBlt'] 

# lets drop original variables
df.drop(["YearBuilt","YearRemodAdd","GarageYrBlt"],1,inplace = True)


# Check variation in the feature values

# In[15]:


# lets first create seperate lists of categorical and numeric columns
cat_vars = []
num_vars = []
for col in df.columns.drop("SalePrice"):
    if df[col].dtypes == 'O':
        cat_vars.append(col)
    else:
        num_vars.append(col)

#lets check the lists created.
print("List of Numeric Columns:",num_vars)
print("\n")
print("List of Categorical Columns:",cat_vars)


# In[16]:


# Let's further seperate the numeric features into continuous and discrete numeric features
num_cont = []
num_disc = []
for col in num_vars:
    if df[col].nunique() > 25: # if variable has more than 25 different values, we consider it as continous variable
        num_cont.append(col)
    else:
        num_disc.append(col)


# In[17]:


# lets check for the variance in the different continuous numeric columns present in the dataset
df.hist(num_cont,bins=50, figsize=(20,15))
plt.tight_layout(pad=0.4)
plt.show()


# In[18]:


# lets check the variance in numbers
for col in num_cont:
    print(df[col].value_counts())
    print("\n")


# Following variables seems to have low variance:
# 
# MasVnArea
# BsmtFinSF2
# 2ndFlrSF
# EnclosedPorch
# ScreenPorch

# In[19]:


# lets check for the variance in the different discrete numeric columns present in the dataset
plt.figure(figsize = (16,96))
for idx,col in enumerate(num_disc):
    plt.subplot(9,2,idx+1)
    ax=sns.countplot(df[col])


# following variables seems to have low variance:
# 
# LowQualFinSF
# BsmtHalfBath
# KitchenAbvGr
# 3SsnPorch
# PoolArea
# MiscVal

# In[20]:


# lets check for the variance in the categorical columns present in the dataset
plt.figure(figsize = (20,200))
for idx,col in enumerate(cat_vars):
    plt.subplot(22,2,idx+1)
    ax=sns.countplot(df[col])
    xticks(rotation=45)


# In[21]:


# lets check the variance in numbers
for col in cat_vars:
    print(df[col].value_counts())
    print("\n")


# Following variables seems to have low variance:
# 
# MSZoning
# Street,
# Alley
# LandContour,
# Utilities,
# LotConfig
# Condition1
# LandSlope
# Condition2,
# BldgType
# RoofStyle
# RoofMatl
# ExterCond
# BsmtCond
# BsmtFinType2
# Heating
# CentralAir
# Electrical
# Functional
# GarageQual
# GarageCond
# PavedDrive
# PoolQC
# Fence
# MiscFeature
# SaleType
# SaleCondition

# In[22]:


# lets drop the variables identified above as they have low variance
low_var_num_cont = ['MasVnrArea','BsmtFinSF2','2ndFlrSF','EnclosedPorch','ScreenPorch']

low_var_num_disc = ['LowQualFinSF','BsmtHalfBath','KitchenAbvGr','3SsnPorch','PoolArea','MiscVal']

low_var_cat_vars = ['MSZoning','Alley','LandContour','Utilities','LotConfig','Condition1','LandSlope','Condition2',
                    'BldgType','RoofStyle','RoofMatl','ExterCond','BsmtCond','BsmtFinType2','Heating','CentralAir',
                    'Electrical','Functional','GarageQual','GarageCond','PavedDrive','PoolQC','SaleType','SaleCondition',
                    'Street','Fence','MiscFeature']

df.drop(low_var_num_cont,1,inplace= True)
df.drop(low_var_num_disc,1,inplace= True)
df.drop(low_var_cat_vars,1,inplace= True)

num_cont = list(set(num_cont)-set(low_var_num_cont))
num_disc = list(set(num_disc)-set(low_var_num_disc))
cat_vars = list(set(cat_vars)-set(low_var_cat_vars))
       
num_vars = num_cont + num_disc


# Lets handle Skewness before moving to Bi-Variate Analysis

# In[23]:


# lets handle skewness in saleprice, lets take log to get normal distribution
df.SalePrice = np.log(df.SalePrice)
 
# lets check the distribution of saleprice again
sns.distplot(df.SalePrice)


# SalePrice looks good now, lets handle other numeric variables

# In[24]:


# taking the log of numeric variables to handle skewness
num_features=['LotFrontage', 'LotArea', '1stFlrSF', 'GrLivArea']
for col in num_features:
    df[col] = np.log(df[col])


# Now we will see how SalePrice varies with respect to "Continous numeric variables" in the dataset

# In[25]:


# now lets plot the graphs for continous variables
plt.figure(figsize=(16,48))
for idx,col in enumerate(num_cont):
    plt.subplot(7,2,idx+1)
    plt.scatter(x = df[col],y=df["SalePrice"])
    plt.ylabel("SalePrice")
    plt.xlabel(col)


# Most of the features above seems to have a good relation with SalePrice
# 
# There are some outliers present which need to be treated
# 
# Now we will see how SalePrice varies with respect to "Discrete numeric variables" in the dataset

# In[26]:


# now lets plot the graphs for discrete variables
plt.figure(figsize=(16,48))
for idx,col in enumerate(num_disc):
    plt.subplot(10,2,idx+1)
    sns.boxplot(x = df[col],y=df["SalePrice"])
    plt.ylabel("SalePrice")
    plt.xlabel(col)


# We can drop MSSubClass, YrSold & MoSold as they have no impact on SalePrice

# In[27]:


# dropping the variables
df.drop(['MSSubClass','YrSold','MoSold'],1,inplace= True)

num_disc = list(set(num_disc)-set(['MSSubClass','YrSold','MoSold']))
num_vars = list(set(num_vars)-set(['MSSubClass','YrSold','MoSold']))


# In[28]:


# lets check relation of sale price with categorical variables
plt.figure(figsize=(16,48))
for idx,col in enumerate(cat_vars):
    plt.subplot(10,2,idx+1)
    sns.boxplot(x = df[col],y=df["SalePrice"])
    xticks(rotation=45)
    plt.ylabel("SalePrice")
    plt.xlabel(col)


# # Outliers Detection

# In[29]:


# lets create boxplots to detect outliars detection 
plt.figure(figsize=(16,48))
for idx,col in enumerate(num_vars):
    plt.subplot(11,2,idx+1)
    plt.boxplot(df[col])
    plt.xlabel(col)


# In[30]:


df.shape


# There are outliers in the dataset, these will be treated in the data engineering section

# In[31]:


for col in num_vars:
    print(df[col].describe(percentiles = [0.05,0.10,0.25,0.50,0.75,0.90,0.95,0.99]))
    print("\n")
    
# lets handle the outliers
q3 = df['OpenPorchSF'].quantile(0.99)
df = df[df.OpenPorchSF <= q3]
    
q3 = df['GarageArea'].quantile(0.99)
df = df[df.GarageArea <= q3]

q3 = df['TotalBsmtSF'].quantile(0.99)
df = df[df.TotalBsmtSF <= q3]

q3 = df['BsmtUnfSF'].quantile(0.99)
df = df[df.BsmtUnfSF <= q3]

q3 = df['WoodDeckSF'].quantile(0.99)
df = df[df.WoodDeckSF <= q3]

q3 = df['BsmtFinSF1'].quantile(0.99)
df = df[df.BsmtFinSF1 <= q3]


# In[32]:


df.shape


# Feature Engineering on Test Data Set

# In[33]:


# lets read the test dataset, we will apply all the feature engineering operations on test set as well
test_set = pd.read_csv("/Users/user/Downloads/all/test.csv")

# save "Id" in a variable and drop the column (as we have already dropped from train dataset)
test_set_id = test_set.Id
test_set.drop("Id",1,inplace = True)

# save SalePrice to a variable and drop it from training dataset as test dataset does not have this column
train_sp = df.SalePrice
df.drop("SalePrice",1,inplace=True)

# all missing values for the categorical columns will be replaced by "None"
# all missing values for the numeric columns will be replaced by median of that field
for col in test_set.columns:
    if test_set[col].dtypes == 'O':
        test_set[col] = test_set[col].replace(np.nan,"None")
    else:
        test_set[col] = test_set[col].replace(np.nan,test_set[col].median())


# creating age of the master from year built to the sale of the master
test_set['HouseAge'] =  test_set['YrSold'] - test_set['YearBuilt']
# age of master after remodelling
test_set['RemodAddAge'] = test_set['YrSold'] - test_set['YearRemodAdd']
# creating age of the garage from year built of the garage to the sale of the master
test_set['GarageAge'] = test_set['YrSold'] - test_set['GarageYrBlt'] 

# lets drop original variables
test_set.drop(["YearBuilt","YearRemodAdd","GarageYrBlt"],1,inplace = True)
        
        
# skewness in test set
# taking the log of numeric variables to hanlde skewness
num_features=['LotFrontage', 'LotArea', '1stFlrSF', 'GrLivArea']
for col in num_features:
    test_set[col] = np.log(test_set[col])

            
test_set.drop(low_var_num_cont,1,inplace= True)
test_set.drop(low_var_num_disc,1,inplace= True)
test_set.drop(low_var_cat_vars,1,inplace= True)

test_set.drop(['MSSubClass','YrSold','MoSold'],1,inplace= True)        
        


    


# In[34]:


test_set.shape


# # To ignore 
# 

# In[ ]:


for col in num_vars:
    print(df[col].describe(percentiles = [0.05,0.10,0.25,0.50,0.75,0.90,0.95,0.99]))
    print("\n")
    
# lets handle the outliers
q3 = test_set['OpenPorchSF'].quantile(0.99)
test_set = test_set[test_set.OpenPorchSF <= q3]
    
q3 = test_set['GarageArea'].quantile(0.99)
test_set = test_set[test_set.GarageArea <= q3]

q3 = test_set['TotalBsmtSF'].quantile(0.99)
test_set = test_set[test_set.TotalBsmtSF <= q3]

q3 = test_set['BsmtUnfSF'].quantile(0.99)
test_set = test_set[test_set.BsmtUnfSF <= q3]

q3 = test_set['WoodDeckSF'].quantile(0.99)
test_set = test_set[test_set.WoodDeckSF <= q3]

q3 = test_set['BsmtFinSF1'].quantile(0.99)
test_set = test_set[test_set.BsmtFinSF1 <= q3]


# In[ ]:


test_set.shape


# In[35]:


# merge the two datasets
master=pd.concat((df,test_set)).reset_index(drop=True)


# In[36]:


master.shape


# In[37]:


# In order to perform linear regression, we need to convert categorical variables to numeric variables.

# We have ordinal variables present in the dataest, lets treat them first:
master['ExterQual'] = master['ExterQual'].map({'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1,'None':0})
master['BsmtQual'] = master['BsmtQual'].map({'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1,'None':0})
master['BsmtExposure'] = master['BsmtExposure'].map({'Gd':4,'Av':3,'Mn':2,'No':1,'None':0})
master['BsmtFinType1'] = master['BsmtFinType1'].map({'GLQ':6,'ALQ':5,'BLQ':4,'Rec':3,'LwQ':2,'Unf':1,'None':0})
master['HeatingQC'] = master['HeatingQC'].map({'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1,'None':0})
master['KitchenQual'] = master['KitchenQual'].map({'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1,'None':0})
master['GarageFinish'] = master['GarageFinish'].map({'Fin':3,'RFn':2,'Unf':1,'None':0})
master['FireplaceQu'] = master['FireplaceQu'].map({'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1,'None':0})


# In[38]:


master.shape


# In[39]:


# now lets create dummy variables for the remaining cateogorical variables
cat_vars = []
for col in master.columns:
    if master[col].dtypes == 'O':
        cat_vars.append(col)

# convert into dummies
master_dummies = pd.get_dummies(master[cat_vars], drop_first=True)

# drop categorical variables 
master.drop(cat_vars,1,inplace = True)

# concat dummy variables with X
master = pd.concat([master, master_dummies], axis=1)

# lets check the shape of the final dataset
master.shape


# In[40]:


# we have perfomed all the necessary operations on the train and test datasets, time to sperate the two sets again
train_set = master[:1372]

test_set = master[1372:]


# In[41]:


train_set


# In[42]:


test_set


# In[43]:


from sklearn.model_selection import train_test_split


# In[44]:


y = train_sp.reset_index(drop=True)
print(y)


# In[45]:


X = train_set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[46]:


XGB = XGBRegressor(max_depth=3,learning_rate=0.1,n_estimators=1000,
                   reg_alpha=0.001,reg_lambda=0.000001,n_jobs=-1,min_child_weight=3)
XGB.fit(X_train,y_train)


# In[47]:


X_train.shape


# In[48]:


X_test.shape


# In[49]:


LGBM = LGBMRegressor(num_leaves=4,max_depth=2,learning_rate=0.01, 
                     n_estimators=2000,max_bin=200,bagging_fraction=0.75,feature_fracton=0.8)
LGBM.fit(X_train,y_train)


# In[50]:


predict_xgb = XGB.predict(X_test)
predict_lgbm = LGBM.predict(X_test)


# In[51]:


metrics.r2_score(y_test, predict_xgb, sample_weight=None, multioutput='uniform_average')


# In[52]:


metrics.r2_score(y_test, predict_lgbm, sample_weight=None, multioutput='uniform_average')


# In[53]:


XGB.fit(X,y)
LGBM.fit(X,y)


# In[57]:


y_pred_xgb  = pd.DataFrame(XGB.predict(test_set))
y_pred_lgbm = pd.DataFrame(LGBM.predict(test_set))

y_pred=pd.DataFrame()
y_pred['SalePrice'] = 0.5 * np.exp(y_pred_xgb[0]) + 0.5 * np.exp(y_pred_lgbm[0])
y_pred['Id'] =test_set_id


# In[59]:


y_pred


# In[60]:


y_pred.to_csv('/Users/user/Downloads/all/sample_submission_final.csv',index=False)


# In[ ]:




