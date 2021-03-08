#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df=pd.read_csv("DCPropertiestrimmed.csv")


# In[ ]:





# In[3]:


df.head() # Prints first n rows of the DataFrame


# In[4]:


df.info() # Index, Datatype and Memory information


# In[ ]:





# In[5]:


df['PRICE'].describe() # Summary statistics for numerical columns


# In[6]:


sns.set(style="whitegrid")
ax = sns.boxplot(x=df["PRICE"])


# In[7]:


ax = sns.boxplot(x= df['PRICE'], y=df['GRADE'], data=df)


# In[8]:


ax = sns.boxplot(x= df['PRICE'], y=df['AC'], data=df)


# In[9]:


ax = sns.boxplot(x= df['PRICE'], y=df['CNDTN'], data=df)


# In[10]:


df['PRICE'].hist(bins=75, rwidth=.8, figsize=(14,4))
plt.title('How expensive are houses?')
plt.xscale('linear')
plt.show()


# In[11]:


sns.boxplot(data= df['PRICE'])


# In[12]:


newAC = df['AC'].replace(to_replace=['N', 'Y'], value=[0, 1])


# In[ ]:





# In[13]:


def remove_outlier(df_in, col_name):
    q1 = df_in[col_name].quantile(0.25)
    q3 = df_in[col_name].quantile(0.75)
    iqr = q3-q1 #Interquartile range
    fence_low  = q1-1.5*iqr
    fence_high = q3+1.5*iqr
    df_out = df_in.loc[(df_in[col_name] > fence_low) & (df_in[col_name] < fence_high)]
    return df_out


# In[14]:


dfn = remove_outlier(df,'PRICE')


# In[15]:


sns.boxplot(data= dfn['PRICE'])


# In[16]:


df2=pd.DataFrame(newAC)


# In[17]:


df['AC']= df2['AC']


# In[18]:


df['AC']


# In[19]:


dfn=df


# In[20]:


dfn


# In[21]:


sns.boxplot(data= df['GBA'])


# In[22]:


sns.boxplot(data= dfn['BEDRM'])


# In[23]:


sns.boxplot(data= dfn['HF_BATHRM'])


# In[24]:


sns.boxplot(data= dfn['LANDAREA'])


# In[25]:


sns.boxplot(data= dfn['Y'])


# In[26]:


dfn = remove_outlier(df,'GBA')
sns.boxplot(data= dfn['GBA'])


# In[27]:


dfn = remove_outlier(df,'LANDAREA')
sns.boxplot(data= df['LANDAREA'])


# In[28]:


dfn['CNDTN']


# In[29]:


def cndtn_to_numeric(x):
    if x=='Excellent':
        return 6
    if x=='Very Good':
        return 5
    if x=='Good':
        return 4
    if x=='Fair':
        return 3
    if x=='Average':
        return 2
    if x=='Poor':
        return 1


# In[30]:


df['CNDTN'] = df['CNDTN'].apply(cndtn_to_numeric)


# In[31]:


dfn=df


# In[32]:


dfn['PRICE'].describe()


# In[33]:


dfn['CNDTN']


# In[34]:


import matplotlib.pyplot as plt


# In[35]:


dfn['EYB'].hist(bins=14, rwidth=.9, figsize=(12,4))
plt.title('When were the houses built?')
plt.show()


# In[36]:


dfn['GBA'].hist(bins=14, rwidth=.9, figsize=(12,4))
plt.title('How big are the building?')
plt.show()


# In[ ]:





# In[37]:


from sklearn.model_selection import train_test_split


# In[38]:


preddf = [dfn['PRICE'], dfn['BATHRM'],dfn['BEDRM'], dfn['HF_BATHRM'],dfn['LANDAREA'],dfn['CNDTN'],dfn['Y'],dfn['FIREPLACES'], dfn['AC'], dfn['ROOMS'], dfn['GBA'], dfn['EYB']]


# In[39]:


preddf=pd.DataFrame(preddf)


# In[40]:


preddf


# In[41]:


preddf = preddf.transpose()


# In[42]:


preddf


# In[43]:


x = preddf[['BATHRM', 'BEDRM','HF_BATHRM','LANDAREA','CNDTN','Y','FIREPLACES','AC','ROOMS','GBA','EYB' ]]
y = preddf['PRICE']


# In[45]:


x


# In[46]:


y


# In[48]:


x_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=101)


# In[49]:


x_train


# In[50]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics


# In[52]:


x_train


# In[53]:


x_train.dropna()


# In[54]:


y_train


# In[55]:


y_train.dropna()


# In[56]:


y_train.mean()


# In[64]:


#Use that outlier formula on xtrain y train and then see


# In[66]:


def clean_dataset(df):
    assert isinstance(df, pd.DataFrame),df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)


# In[ ]:





# In[69]:


clean_dataset(df)


# In[70]:


df


# In[71]:


df['PRICE'].dropna()


# In[91]:


df=pd.read_csv("DCPropertiestrimmed.csv")


# In[92]:


df=df.dropna()


# In[93]:


df


# In[94]:


to_drop = ['X','Y','SOURCE','BLDG_NUM','STATE']


# In[95]:


df=df.dropcolumn(to_drop, inplace=True, axis=1)


# In[96]:


df.drop(columns=to_drop, inplace=True)


# In[97]:


df


# In[101]:


from scipy import stats
import numpy as np
z = np.abs(stats.zscore(df['PRICE']))
print(z)


# In[106]:


df['PRICE']


# In[112]:


df['PRICE']


# In[113]:


numeric_data = df.select_dtypes(include=[np.number])
categorical_data = df.select_dtypes(exclude=[np.number])


# In[ ]:





# In[114]:


numeric_data


# In[115]:


categorical_data


# In[116]:


from scipy import stats
numeric_data[(np.abs(stats.zscore(numeric_data)) < 3).all(axis=1)]


# In[ ]:





# In[117]:


numdata = df.select_dtypes(include=[np.number])
catdata = df.select_dtypes(exclude=[np.number])


# In[118]:


numdata


# In[120]:


Q1 = numdata.quantile(0.02)
Q3 = numdata.quantile(0.98)
IQR = Q3 - Q1
numdata = numdata[~((numdata < (Q1 - 1.5 * IQR)) |(numdata > (Q3 + 1.5 * 
IQR))).any(axis=1)]


# In[121]:


numdata


# In[122]:


def cndtn_to_numeric(x):
    if x=='Excellent':
        return 6
    if x=='Very Good':
        return 5
    if x=='Good':
        return 4
    if x=='Fair':
        return 3
    if x=='Average':
        return 2
    if x=='Poor':
        return 1


# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df=pd.read_csv("DCPropertiestrimmed.csv")


# In[3]:


df['STYLE']=pd.get_dummies(df['STYLE'])


# In[4]:


df['EXTWALL']=pd.get_dummies(df['EXTWALL'])


# In[16]:


df['ROOF']=pd.get_dummies(df['ROOF'])


# In[17]:


df['INTWALL']=pd.get_dummies(df['INTWALL'])


# In[18]:


df['WARD']=pd.get_dummies(df['WARD'])


# In[19]:


df


# In[22]:


df=pd.get_dummies(df, columns=['WARD']).head()


# In[21]:


df


# In[9]:


df_sorted= df.sort_values('PRICE')
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure(figsize=(20,10))
sns.barplot(y='PRICE', x='GRADE', data=df)
sns.set(font_scale=1) 
plt.xticks(rotation=45)

