#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[1]:


df=pd.read_csv("./Desktop/crx.data")


# In[3]:


df.head()


# In[4]:


df.tail()


# In[5]:


newAC = df['AC'].replace(to_replace=['N', 'Y'], value=[0, 1])


# In[6]:


print(newAC)


# In[7]:


df2=pd.DataFrame(newAC)


# In[8]:


df2


# In[9]:


df['AC']= df2['AC']


# In[10]:


df.head()


# In[1]:


df['PRICE'].plot(kind='hist', xlabelsize=12, ylabelsize=12);


# In[12]:


df.drop(columns=['FULLADDRESS','CITY','STATE'])


# In[13]:


df.dropna()


# In[14]:


df= df.drop(columns=['FULLADDRESS','CITY', 'STATE', 'ZIPCODE', 'LATITUDE', 'LONGITUDE', 'ASSESSMENT_NBHD', 'ASSESSMENT_SUBNBHD'])


# In[15]:


df= df.dropna()


# In[16]:


df


# In[17]:


df= df.drop(columns=['SOURCE'])


# In[18]:


df


# In[19]:


def remove_outlier(df_in, col_name):
    q1 = df_in[col_name].quantile(0.25)
    q3 = df_in[col_name].quantile(0.75)
    iqr = q3-q1 #Interquartile range
    fence_low  = q1-1.5*iqr
    fence_high = q3+1.5*iqr
    df_out = df_in.loc[(df_in[col_name] > fence_low) & (df_in[col_name] < fence_high)]
    return df_out


# In[20]:


dfn = remove_outlier(df,'PRICE')


# In[21]:


dfn


# In[22]:


df


# In[23]:


dfn = remove_outlier(df,'PRICE')


# In[24]:


dfn['PRICE']


# In[25]:


from pandas.plotting import scatter_matrix


# In[26]:


import plotly.express as px
fig = px.scatter_matrix(dfn,dimensions=["PRICE", "BATHRM","AC", "NUM_UNITS", "ROOMS", "BEDRM","GBA","KITCHENS", "FIREPLACES", "LANDAREA"], 
                       )
fig.show()


# In[27]:


import plotly.express as px
fig = px.scatter_matrix(dfn,
    dimensions=["PRICE", "BATHRM","AC", "NUM_UNITS", "ROOMS", "BEDRM","GBA","KITCHENS", "FIREPLACES", "LANDAREA"],
    
    title="Scatter matrix",
    labels={col:col.replace('_', ' ') for col in dfn.columns}) # remove underscore
fig.update_traces(diagonal_visible=False)
fig.update_yaxes(automargin=True)
fig.update_xaxes(automargin=True)


fig.show()


# In[28]:


# Feature sorted by correlation to PRICE, from positive to negative
corr = dfn.corr()
corr = corr.sort_values('PRICE', ascending=False)
plt.figure(figsize=(8,10))
sns.barplot( corr.PRICE[1:], corr.index[1:], orient='h')
plt.show()


# In[29]:


#cat_features = dfn.select_dtypes(include=['object']).columns


# In[30]:


dfn['PRICE'].hist(bins=75, rwidth=.8, figsize=(14,4))
plt.title('How expensive are houses?')
plt.show()


# In[31]:


plt.figure(figsize=(10,6))
sns.distplot(dfn.PRICE)
plt.title('How expensive are houses?')
plt.show()


# In[81]:


num_features = dfn.select_dtypes(include=['int64','float64']).columns


# In[84]:


ab= sns.boxplot(data=num_features,orient='h', palette='Set3')


# In[33]:


# Grid of distribution plots of all numerical features
f = pd.melt(dfn, value_vars=sorted(num_features))
g = sns.FacetGrid(f, col='variable', col_wrap=4, sharex=False, sharey=False)
g = g.map(sns.distplot, 'value')


# In[39]:


corr = dfn.corr()
plt.figure(figsize=(50,15))
ax = sns.heatmap(
    corr, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
);


# In[93]:


#It's a nice overview, but oh man is that a lot of data to look at. 
#Let's zoom into the top 10 features most related to Sale Price.
# Top 10 Heatmap

plt.figure(figsize=(50,15))
k = 10 #number of variables for heatmap
cols = corr.nlargest(k, 'PRICE')['PRICE'].index
cm = np.corrcoef(df[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 15}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()
plt.xticks(rotation=45)


# In[94]:


most_corr = pd.DataFrame(cols)
most_corr.columns = ['Most Correlated Features']
most_corr


# In[73]:


plt.figure(figsize=(50,25))
corrMatrix = dfn.corr()
sns.heatmap(corrMatrix, annot=True, annot_kws={"size":30})
plt.show()


# In[ ]:





# In[ ]:


print( dfn.corr())


# In[ ]:


dfn[['PRICE','BATHRM','AC', 'SQUARE', 'ROOMS', 'BEDRM','GBA', 'EYB', 'HF_BATHRM', 'GRADE', 'LANDAREA', 'KITCHENS', 'FIREPLACES','EYB', 'YR_RMDL','GRADE','Y']].corr()


# In[44]:


plt.figure(figsize=(50,15))
sns.jointplot(x='PRICE',y = 'BATHRM', kind = 'hex', data = dfn)


# In[96]:


sns.jointplot(x=df['GBA'], y=df['PRICE'], kind='reg')


# In[99]:


sns.jointplot(x=df['LANDAREA'], y=df['PRICE'], kind='reg')


# In[100]:


dfn.info()


# In[45]:


fig, ax = plt.subplots()
ax.boxplot(dfn)


# title and axis labels
ax.set_title('box plot')
ax.set_xlabel('x-axis')
ax.set_ylabel('y-axis')
xticklabels=['PRICE','BATHRM','AC', 'NUM_UNITS', 'ROOMS', 'BEDRM','GBA', 'EYB', 'HF_BATHRM', 'LANDAREA', 'KITCHENS', 'FIREPLACES','EYB', 'YR_RMDL',]
ax.set_xticklabels(xticklabels)


# add horizontal grid lines
ax.yaxis.grid(True)


# show the plot
plt.show()


# In[46]:


sns.boxplot(data= dfn['PRICE'])


# In[ ]:


datan= [df['PRICE'], df['BATHRM'],  df['NUM_UNITS'], df['BEDRM'], df['HF_BATHRM'], df['LANDAREA'], df['KITCHENS'], df['FIREPLACES']]


# In[47]:


sns.boxplot(data= dfn['BATHRM'])


# In[ ]:


dfn = remove_outlier(df,'BATHRM')


# In[ ]:


sns.boxplot(data= dfn['BATHRM'])


# In[ ]:


sns.boxplot(data= dfn['BEDRM'])


# In[ ]:


dfn = remove_outlier(df,'BEDRM')


# In[ ]:


sns.boxplot(data= dfn['BEDRM'])


# In[ ]:


sns.boxplot(data= dfn['HF_BATHRM'])


# In[ ]:


dfn = remove_outlier(df,'HF_BATHRM')


# In[ ]:


sns.boxplot(data= dfn['HF_BATHRM'])


# In[ ]:


sns.boxplot(data= dfn['LANDAREA'])


# In[ ]:


dfn = remove_outlier(df,'LANDAREA')


# In[ ]:


sns.boxplot(data= dfn['LANDAREA'])


# In[ ]:


sns.boxplot(data= dfn['KITCHENS'])


# In[ ]:


dfn = remove_outlier(df,'KITCHENS')


# In[ ]:





# In[ ]:


sns.boxplot(data= dfn['FIREPLACES'])


# In[ ]:


dfn = remove_outlier(df,'FIREPLACES')


# In[ ]:





# In[ ]:


sns.boxplot(data= dfn['X'])


# In[ ]:


sns.boxplot(data= dfn['EYB'])


# In[ ]:





# In[ ]:


sns.boxplot(data= dfn['GBA'])


# In[ ]:


dfn = remove_outlier(df,'GBA')


# In[ ]:


sns.boxplot(data= dfn['GBA'])


# In[ ]:


sns.boxplot(data= dfn['SQUARE'])


# In[ ]:


dfn


# In[ ]:


newcndn = df['CNDTN'].replace(to_replace=['Average', 'Excellent, 'Fair', 'Good', 'Poor', 'Very Good'], value=[2, 6, 3, 4, 1, 5])


# In[ ]:


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


# In[ ]:





# In[ ]:


df['CNDTN']


# In[ ]:


dfn['CNDTN'] = dfn['CNDTN'].apply(cndtn_to_numeric)


# In[ ]:


dfn['CNDTN']


# In[ ]:


dfn['CNDTN']


# In[ ]:


#df['A'].corr(df['B'])

dfn['PRICE'].corr(dfn['CNDTN'])


# In[ ]:


dfn['PRICE'].corr(dfn['CENSUS_TRACT'])


# In[ ]:


datan= [df['PRICE'], df['BATHRM'], 
        df['BEDRM'], df['HF_BATHRM'],
        df['LANDAREA'],df['CNDTN'],df['Y'], 
        df['FIREPLACES'], df['AC'], df['ROOMS'], df['GBA'], df['EYB']]


# In[ ]:


datan=pd.DataFrame(datan)


# In[ ]:


corr = datan.corr()
ax = sns.heatmap(
    corr, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
);


# In[ ]:


preddf=pd.DataFrame(preddf)


# In[ ]:


preddf = preddf.transpose()


# In[ ]:


preddf


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix,accuracy_score


# In[ ]:





# In[104]:


plt.figure(figsize=(6,8))


# In[107]:


plt.figure(figsize=(12,6))
ab= sns.boxplot(data=df[['BATHRM','BEDRM','HF_BATHRM','FIREPLACES','ROOMS']],orient='h', palette='Set3')


# In[ ]:





# In[ ]:


ab= sns.boxplot(data=df[['AYB']],orient='h', palette='Set3')


# In[ ]:


ab= sns.boxplot(data=df[['EYB']],orient='h', palette='Set3')


# In[ ]:


ab= sns.boxplot(data=df[['YR_RMDL']],orient='h', palette='Set3')


# In[48]:


plt.figure(figsize=(50,15))
sns.set(font_scale=3) 
ax = sns.countplot(x="STYLE", data=dfn)
plt.xticks(rotation=45)


# In[49]:


plt.figure(figsize=(50,15))
sns.set(font_scale=3) 
ax = sns.countplot(x="CNDTN", data=dfn)
plt.xticks(rotation=45)


# In[51]:


plt.figure(figsize=(50,15))
sns.set(font_scale=3) 
ax = sns.countplot(x="STRUCT", data=dfn)
plt.xticks(rotation=45)


# In[50]:


plt.figure(figsize=(50,15))
sns.set(font_scale=3) 
ax = sns.countplot(x="EXTWALL", data=dfn)
plt.xticks(rotation=45)


# #fig=plt.figure(figsize=(50,15))
# #ax1=fig.add_subplot(221)
# #sns.barplot(x=df['PRICE'], y=df['BATHRM'])

# In[62]:


plt.figure(figsize=(20,10))
sns.barplot(y='PRICE', x='CNDTN', data=dfn)
sns.set(font_scale=2) 
plt.xticks(rotation=45)


# In[63]:


plt.figure(figsize=(20,10))
sns.barplot(y='PRICE', x='EXTWALL', data=dfn)
sns.set(font_scale=2) 
plt.xticks(rotation=45)


# In[64]:


plt.figure(figsize=(20,10))
sns.barplot(y='PRICE', x='ROOF', data=dfn)
sns.set(font_scale=2) 
plt.xticks(rotation=45)


# In[65]:


plt.figure(figsize=(20,10))
sns.barplot(y='PRICE', x='INTWALL', data=dfn)
sns.set(font_scale=2) 
plt.xticks(rotation=45)


# In[74]:


plt.figure(figsize=(20,10))
sns.boxplot(y='PRICE', x='INTWALL', data=dfn)
sns.set(font_scale=2) 
plt.xticks(rotation=45)


# In[77]:


plt.figure(figsize=(20,10))
sns.boxplot(y='PRICE', x='CNDTN', data=dfn)
sns.set(font_scale=2) 
plt.xticks(rotation=45)


# In[80]:


plt.figure(figsize=(20,10))
sns.boxplot(y='PRICE', x='ROOF', data=dfn)
sns.set(font_scale=2) 
plt.xticks(rotation=45)


# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:





# In[3]:


df=pd.read_csv("DCPropertiestrimmed.csv")


# In[4]:


df=pd.get_dummies(df, columns=['WARD']).head()


# In[5]:


df=pd.get_dummies(df, columns=['CNDTN']).head()


# In[7]:


df=pd.get_dummies(df, columns=['EXTWALL']).head()


# In[8]:


df=pd.get_dummies(df, columns=['INTWALL']).head()


# In[9]:


df=pd.get_dummies(df, columns=['STYLE']).head()


# In[12]:


df.info()


# In[14]:


df.info()


# In[21]:


df


# In[36]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[37]:


df=pd.read_csv("DCPropertiestrimmed.csv")


# In[38]:


df=pd.get_dummies(df, columns=['INTWALL'])


# In[39]:


df


# In[40]:


df=pd.get_dummies(df, columns=['EXTWALL'])


# In[41]:


df=pd.get_dummies(df, columns=['ROOF'])


# In[42]:


df=pd.get_dummies(df, columns=['WARD'])


# In[43]:


df


# In[44]:


df=pd.get_dummies(df, columns=['STRUCT'])


# In[45]:


df=pd.get_dummies(df, columns=['STYLE'])


# In[46]:


df=pd.get_dummies(df, columns=['QUADRANT'])


# In[47]:


df


# In[48]:


df.dropna()


# In[49]:


df= df.drop(columns=['FULLADDRESS','CITY', 'STATE', 'ZIPCODE', 'LATITUDE', 'LONGITUDE', 'ASSESSMENT_NBHD', 'ASSESSMENT_SUBNBHD'])


# In[50]:


df


# In[51]:


df.info()


# In[4]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure(figsize=(20,10))
sns.barplot(y='PRICE', x='GRADE', data=df)
sns.set(font_scale=2) 
plt.xticks(rotation=45)


# In[ ]:


#Converting categorical into numerical variables 

#sex = pd.get_dummies(train['Sex'],drop_first=True)
#embark = pd.get_dummies(train['Embarked'],drop_first=True)
#Here, we are dummying the sex and embark columns. After dummying, we will drop the rest of the columns which are not needed.
#train.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
#We will concatenate the new sex and embarked columns to the dataframe.
#train = pd.concat([train,sex,embark],axis=1)

