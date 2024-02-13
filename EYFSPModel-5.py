#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Check free memory available
get_ipython().run_line_magic('system', 'free -m')


# In[2]:


#! pip install pandas_gbq


# 
# ![Screenshot 2024-01-02 at 09.53.29.png]
# 

# In[3]:


# Import required libraries
from google.cloud import bigquery
import gc
from dateutil.relativedelta import relativedelta
import numpy as np
import math
import os
import pandas_gbq as pdg
import pandas as pd
# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)


# <div class="alert alert-block alert-info">
#     
# create or replace table `yhcr-prd-phm-bia-core.CB_MYSPACE_RT.EarlyYears` as 
# SELECT person_id,AcademicYear,AcademicAge,Gender,PSEAS1,PSEAS2,PSEAS3,PSETotal,
#     CLLAS1,CLLAS2,CLLAS3,CLLAS4,CLLTotal,
#     PSRNAS1,PSRNAS2,PSRNAS3,PSRNTotal,
#     RKUW,RIPD,RICD,EYFSPTotal,
#     COMG01,COMG02,COMG03,PHYG04,PHYG05,PSEG06,PSEG07,PSEG08,
#     LITG09,LITG10,MATG11, MATG12,UTWG13,UTWG14,UTWG15,EXPG16,EXPG17,GLD FROM `yhcr-prd-phm-bia-core.CB_FDM_DepartmentForEducation.src_EYFSP` ;
# 
# </div>

# In[4]:


# Instaniate BigQuery client

sqlEY = """ SELECT person_id,AcademicYear,Gender,PSEAS1,PSEAS2,PSEAS3,PSETotal,
    CLLAS1,CLLAS2,CLLAS3,CLLAS4,CLLTotal,
    PSRNAS1,PSRNAS2,PSRNAS3,PSRNTotal,
    RKUW,RIPD,RICD,EYFSPTotal,
    COMG01,COMG02,COMG03,PHYG04,PHYG05,PSEG06,PSEG07,PSEG08,
    LITG09,LITG10,MATG11, MATG12,UTWG13,UTWG14,UTWG15,EXPG16,EXPG17,GLD  FROM `yhcr-prd-phm-bia-core.CB_FDM_DepartmentForEducation.src_EYFSP` a """

EarlyYears = pdg.read_gbq(sqlEY, dialect='standard')


# In[5]:


df = EarlyYears


# In[6]:


len(df)


# In[7]:


df.person_id.nunique()
# there are 130551 unique records in EYFS table
# 89 records as duplicates


# In[8]:


df = df.drop_duplicates(subset=['person_id'])
len(df)


# In[9]:


df = df.copy(deep=True)


# <div class="alert alert-block alert-info">
# <h2> 
# Gender - Aggregate all records with 'F' and 'f' as 'F' - Female
# All 'M' and 'm' as 'M' - Male
# </h2>
# </div>

# In[10]:


df['newGLD'] = df['GLD'].apply(lambda set_: False if pd.isna(set_)== True else set_)
df['Gender'] = df['Gender'].apply(lambda set_: 'F' if (set_== 'f') else set_ )
df['Gender'] = df['Gender'].apply(lambda set_: 'M' if (set_== 'm') else set_)


# In[11]:


df['newGLD'] = df['GLD'].apply(lambda set_: False if pd.isna(set_)== True else set_)
df['Gender'] = df['Gender'].apply(lambda set_: 'F' if (set_== 'f') else set_ )
df['Gender'] = df['Gender'].apply(lambda set_: 'M' if (set_== 'm') else set_)
df["AcademicBegin"]  = df["AcademicYear"].str.slice(0, 4)
df["AcademicEnd"] = df["AcademicYear"].str.slice(5)
#df


# <div class="alert alert-block alert-info">
# <h2> Changing all scores datatype to integer for calculations purposes from string and handling None
# </h2>
# </div>

# In[12]:


#df['PSEAS1'] = df['PSEAS1'].apply(lambda set_: 0 if np.nan(set_)== True else int(set_))

df['PSEAS1'] = df['PSEAS1'].replace({None: 0,'N': 0})  
df['PSEAS2'] = df['PSEAS2'].replace({None: 0,'N': 0}) 
df['PSEAS3'] = df['PSEAS3'].replace({None: 0,'N': 0})
df['PSETotal'] = df['PSETotal'].replace({None: 0,'N': 0})
df['CLLAS1'] = df['CLLAS1'].replace({None: 0,'N': 0})
df['CLLAS2'] = df['CLLAS2'].replace({None: 0,'N': 0})
df['CLLAS3'] = df['CLLAS3'].replace({None: 0,'N': 0})
df['CLLAS4'] = df['CLLAS4'].replace({None: 0,'N': 0})
df['CLLTotal'] = df['CLLTotal'].replace({None: 0,'N': 0})
df['PSRNAS1'] = df['PSRNAS1'].replace({None: 0,'N': 0})
df['PSRNAS2'] = df['PSRNAS2'].replace({None: 0,'N': 0})
df['PSRNAS3'] = df['PSRNAS3'].replace({None: 0,'N': 0})
df['PSRNTotal'] = df['PSRNTotal'].replace({None: 0,'N': 0})
df['RKUW'] = df['RKUW'].replace({None: 0,'N': 0})
df['RIPD'] = df['RIPD'].replace({None: 0,'N': 0})
df['RICD'] = df['RICD'].replace({None: 0,'N': 0})
df['EYFSPTotal'] = df['EYFSPTotal'].replace({None: 0,'N': 0})

df['PSEAS1'] = df['PSEAS1'].astype(int)
df['PSEAS2'] = df['PSEAS2'].astype(int)
df['PSEAS3'] = df['PSEAS3'].astype(int)
df['PSETotal'] = df['PSETotal'].astype(int)
df['CLLAS1'] = df['CLLAS1'].astype(int)
df['CLLAS2'] = df['CLLAS2'].astype(int)
df['CLLAS3'] = df['CLLAS3'].astype(int)
df['CLLAS4'] = df['CLLAS4'].astype(int)
df['CLLTotal'] = df['CLLTotal'].astype(int)
df['PSRNAS1'] = df['PSRNAS1'].astype(int)
df['PSRNAS2'] = df['PSRNAS2'].astype(int)
df['PSRNAS3'] = df['PSRNAS3'].astype(int)
df['PSRNTotal'] = df['PSRNTotal'].astype(int)
df['RKUW'] = df['RKUW'].astype(int)
df['RIPD'] = df['RIPD'].astype(int)
df['RICD'] = df['RICD'].astype(int)
df['EYFSPTotal'] = df['EYFSPTotal'].astype(int)
df['AcademicBegin'] = df['AcademicBegin'].astype(int)
df['AcademicEnd'] = df['AcademicEnd'].astype(int)

#df.dtypes
                                     


# <div class="alert alert-block alert-info">
# <h3> For records pertaining to early years from 2002 - 2012 we have to set the GLD flag based on computation <br/>
#          1. GLD Flag set it to true if all individual learning goals is above or equal to 6 and <br/>
#          2. if the Total is greater than or equal to 78. Otherwise the GLD flag is set to false <br/>
# </h3>
# </div>

# In[13]:


df.loc[(df['PSEAS1']>=6)&(df['PSEAS2']>=6)&(df['PSEAS3']>=6)&
       (df['CLLAS1']>=6)&(df['CLLAS2']>=6)&(df['CLLAS3']>=6)&(df['CLLAS4']>=6)&
       (df['PSRNAS1']>=6)&(df['PSRNAS2']>=6)&(df['PSRNAS3']>=6)&
       (df['RKUW']>=6)&(df['RICD']>=6)&(df['RIPD']>=6)&(df['EYFSPTotal']>=78), 'newGLD'] = True   


# In[14]:


rf = df
rf


# In[15]:


#! pip  install hvplot
#! pip  install xarray


# In[16]:


import hvplot.pandas  # noqa
import xarray as xr
import pandas as pd
import panel as pn


# In[17]:


rf = rf.drop(['GLD'],axis=1)


# In[18]:


dfinteractive = rf.interactive()


# In[19]:


year_slider = pn.widgets.IntSlider(name='Academic Year slider', start=2002, end=2019, step=1, value=2019)
year_slider


# In[20]:


# yaxis_NCCISCODE_source = 'person_id'

# yaxis_NCCIS_source = 'EYFSPTotal'

# NCCIS_source_bar_pipeline = (
#     dfinteractive[
#         (dfinteractive.AcademicBegin <= year_slider)
#     ]
#     .groupby(['AcademicBegin','Gender'])[yaxis_NCCISCODE_source].nunique()
#     .to_frame()
#     .reset_index()
#     .sort_values(by='AcademicBegin')  
#     .reset_index(drop=True)
# )
# NCCIS_plot = NCCIS_source_bar_pipeline.hvplot(kind='bar',stacked=True,legend="top_left",height=500,
#                                                      x='AcademicBegin',by='Gender',y=yaxis_NCCISCODE_source, 
#                                                      title='Gender appearance per Academic Year in EYFSP',width=1000)

# NCCIS_plot


# In[21]:


EYGP = rf.groupby(['AcademicBegin','Gender', 'newGLD']).agg({
                    'Gender':'value_counts'
}).rename(columns={'Gender':'COUNTByGender'})
EYGP


# In[22]:


matrix = EYGP.reset_index()
GLDTrueFGenderData = matrix[(matrix['newGLD']==True) & (matrix['Gender']=='F') ]
GLDTrueMGenderData = matrix[(matrix['newGLD']==True) & (matrix['Gender']=='M') ]

GLDFalseFGenderData = matrix[(matrix['newGLD']==False) & (matrix['Gender']=='F') ]
GLDFalseMGenderData = matrix[(matrix['newGLD']==False) & (matrix['Gender']=='M') ]

GLDTrueFGenderData = GLDTrueFGenderData.drop(['Gender','newGLD'], axis=1)
GLDTrueMGenderData = GLDTrueMGenderData.drop(['Gender','newGLD'], axis=1)

GLDFalseFGenderData = GLDFalseFGenderData.drop(['Gender','newGLD'], axis=1)
GLDFalseMGenderData = GLDFalseMGenderData.drop(['Gender','newGLD'], axis=1)


# In[23]:


from matplotlib import ticker
import matplotlib.pyplot as plt
plt.figure(figsize=(30,30))

# plt.bar(X_axis - 0.2, GLDTrueFGenderData, 0.4, color="purple",label = 'Women Passed GLD')
# plt.bar(X_axis + 0.2, GLDTrueMGenderData, 0.4, color="red", label = 'Men Passed GLD')
  
ax = GLDTrueFGenderData.set_index('AcademicBegin').plot.bar(color="lightgreen")
bx = GLDTrueMGenderData.set_index('AcademicBegin').plot.bar(color="lightblue")


ax.set_ylabel("Count of Women who passed GLD")
bx.set_ylabel("Count of Men who passed GLD")

#plt.xticks(X_axis, X_Label)
plt.legend()
plt.xticks(rotation = 90, fontsize = 10)
plt.xlabel("Academic Years")

plt.title("GLD Attainment")


for bar in ax.patches:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2.0, height, f' {height:.2f}', fontsize=10,
            ha='center')
    ax.set_title("Count of Women who passed GLD")    
    
for bar in bx.patches:
    height = bar.get_height()
    bx.text(bar.get_x() + bar.get_width() / 2.0, height, f' {height:.2f}', fontsize=10,
            ha='center')
    bx.set_title("Count of Men who passed GLD")    

plt.show()

 


# In[24]:


from matplotlib import ticker
import matplotlib.pyplot as plt
plt.figure(figsize=(30,30))

# plt.bar(X_axis - 0.2, GLDTrueFGenderData, 0.4, color="purple",label = 'Women Passed GLD')
# plt.bar(X_axis + 0.2, GLDTrueMGenderData, 0.4, color="red", label = 'Men Passed GLD')
  
ax = GLDFalseFGenderData.set_index('AcademicBegin').plot.bar(color="lightpink")
bx = GLDFalseMGenderData.set_index('AcademicBegin').plot.bar(color="yellow")


ax.set_ylabel("Count Female Students Not Attaining GLD")
bx.set_ylabel("Count Male Students Not Attaining GLD")

#plt.xticks(X_axis, X_Label)
plt.legend()
plt.xticks(rotation = 90, fontsize = 10)
plt.xlabel("Academic Years")

plt.title("GLD Attainment")


for bar in ax.patches:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2.0, height, f' {height:.2f}', fontsize=10,
            ha='center')
    ax.set_title("Count Female Students Not Attaining GLD")    
    
for bar in bx.patches:
    height = bar.get_height()
    bx.text(bar.get_x() + bar.get_width() / 2.0, height, f' {height:.2f}', fontsize=10,
            ha='center')
    bx.set_title("Count Male Students Not Attaining GLD")    

plt.show()

 


# <div class="alert alert-block alert-info">
# <h3> For records pertaining to early years from 2002 - 2018 we have to retreive data fro the NEET Summary Table<br/>
#          1. Analyse the records <br/>
#          2. In compliance with the Child Act of 2006, we will exclude data from 2002 to 2005. Subsequently, we will cross-reference records between EYFS and NCCIS, focusing on students aged 16-18 years. <br/>
# </h3>
# </div>

# In[25]:


# Instaniate BigQuery client

sqlEYFSNEET = """ SELECT a.person_id,a.AcademicYear,a.Gender,a.PSEAS1,a.PSEAS2,a.PSEAS3,a.PSETotal,
    a.CLLAS1,a.CLLAS2,a.CLLAS3,a.CLLAS4,a.CLLTotal,
    a.PSRNAS1,a.PSRNAS2,a.PSRNAS3,a.PSRNTotal,
    a.RKUW,a.RIPD,a.RICD,a.EYFSPTotal,
    a.COMG01,a.COMG02,a.COMG03,a.PHYG04,a.PHYG05,a.PSEG06,a.PSEG07,a.PSEG08,
    a.LITG09,a.LITG10,a.MATG11, a.MATG12,a.UTWG13,a.UTWG14,a.UTWG15,a.EXPG16,a.EXPG17,a.GLD, b.* 
FROM `yhcr-prd-phm-bia-core.CB_FDM_DepartmentForEducation.src_EYFSP` a, `yhcr-prd-phm-bia-core.CB_2166.wide_format_NEET_final` b where a.person_id = b.person_id """

EYFSPDF = pdg.read_gbq(sqlEYFSNEET, dialect='standard')
#sqlWideFormat


# In[26]:


EYFSPDF


# In[27]:


EYSFTransactDF = EYFSPDF
len(EYSFTransactDF)


# In[28]:


EYSFTransactDF = EYSFTransactDF.drop_duplicates(subset=['person_id'])
len(EYSFTransactDF)


# In[29]:


EYSFTransactDF = EYSFTransactDF.copy(deep=True)


# In[30]:


EYSFTransactDF['newGLD'] = EYSFTransactDF['GLD'].apply(lambda set_: False if pd.isna(set_)== True else set_)
EYSFTransactDF['Gender'] = EYSFTransactDF['Gender'].apply(lambda set_: 'F' if (set_== 'f') else set_ )
EYSFTransactDF['Gender'] = EYSFTransactDF['Gender'].apply(lambda set_: 'M' if (set_== 'm') else set_)

EYSFTransactDF['newGLD'] = EYSFTransactDF['GLD'].apply(lambda set_: False if pd.isna(set_)== True else set_)
EYSFTransactDF['Gender'] = EYSFTransactDF['Gender'].apply(lambda set_: 'F' if (set_== 'f') else set_ )
EYSFTransactDF['Gender'] = EYSFTransactDF['Gender'].apply(lambda set_: 'M' if (set_== 'm') else set_)
EYSFTransactDF["AcademicBegin"]  = EYSFTransactDF["AcademicYear"].str.slice(0, 4)
EYSFTransactDF["AcademicEnd"] = EYSFTransactDF["AcademicYear"].str.slice(5)
#df


# In[31]:


EYSFTransactDF['PSEAS1'] = EYSFTransactDF['PSEAS1'].replace({None: 0,'N': 0})  
EYSFTransactDF['PSEAS2'] = EYSFTransactDF['PSEAS2'].replace({None: 0,'N': 0}) 
EYSFTransactDF['PSEAS3'] = EYSFTransactDF['PSEAS3'].replace({None: 0,'N': 0})
EYSFTransactDF['PSETotal'] = EYSFTransactDF['PSETotal'].replace({None: 0,'N': 0})
EYSFTransactDF['CLLAS1'] = EYSFTransactDF['CLLAS1'].replace({None: 0,'N': 0})
EYSFTransactDF['CLLAS2'] = EYSFTransactDF['CLLAS2'].replace({None: 0,'N': 0})
EYSFTransactDF['CLLAS3'] = EYSFTransactDF['CLLAS3'].replace({None: 0,'N': 0})
EYSFTransactDF['CLLAS4'] = EYSFTransactDF['CLLAS4'].replace({None: 0,'N': 0})
EYSFTransactDF['CLLTotal'] = EYSFTransactDF['CLLTotal'].replace({None: 0,'N': 0})
EYSFTransactDF['PSRNAS1'] = EYSFTransactDF['PSRNAS1'].replace({None: 0,'N': 0})
EYSFTransactDF['PSRNAS2'] = EYSFTransactDF['PSRNAS2'].replace({None: 0,'N': 0})
EYSFTransactDF['PSRNAS3'] = EYSFTransactDF['PSRNAS3'].replace({None: 0,'N': 0})
EYSFTransactDF['PSRNTotal'] = EYSFTransactDF['PSRNTotal'].replace({None: 0,'N': 0})
EYSFTransactDF['RKUW'] = EYSFTransactDF['RKUW'].replace({None: 0,'N': 0})
EYSFTransactDF['RIPD'] = EYSFTransactDF['RIPD'].replace({None: 0,'N': 0})
EYSFTransactDF['RICD'] = EYSFTransactDF['RICD'].replace({None: 0,'N': 0})
EYSFTransactDF['EYFSPTotal'] = EYSFTransactDF['EYFSPTotal'].replace({None: 0,'N': 0})


EYSFTransactDF['PSEAS1'] = EYSFTransactDF['PSEAS1'].astype(int)
EYSFTransactDF['PSEAS2'] = EYSFTransactDF['PSEAS2'].astype(int)
EYSFTransactDF['PSEAS3'] = EYSFTransactDF['PSEAS3'].astype(int)
EYSFTransactDF['PSETotal'] = EYSFTransactDF['PSETotal'].astype(int)
EYSFTransactDF['CLLAS1'] = EYSFTransactDF['CLLAS1'].astype(int)
EYSFTransactDF['CLLAS2'] = EYSFTransactDF['CLLAS2'].astype(int)
EYSFTransactDF['CLLAS3'] = EYSFTransactDF['CLLAS3'].astype(int)
EYSFTransactDF['CLLAS4'] = EYSFTransactDF['CLLAS4'].astype(int)
EYSFTransactDF['CLLTotal'] = EYSFTransactDF['CLLTotal'].astype(int)
EYSFTransactDF['PSRNAS1'] = EYSFTransactDF['PSRNAS1'].astype(int)
EYSFTransactDF['PSRNAS2'] = EYSFTransactDF['PSRNAS2'].astype(int)
EYSFTransactDF['PSRNAS3'] = EYSFTransactDF['PSRNAS3'].astype(int)
EYSFTransactDF['PSRNTotal'] = EYSFTransactDF['PSRNTotal'].astype(int)
EYSFTransactDF['RKUW'] = EYSFTransactDF['RKUW'].astype(int)
EYSFTransactDF['RIPD'] = EYSFTransactDF['RIPD'].astype(int)
EYSFTransactDF['RICD'] = EYSFTransactDF['RICD'].astype(int)
EYSFTransactDF['EYFSPTotal'] = EYSFTransactDF['EYFSPTotal'].astype(int)
EYSFTransactDF['AcademicBegin'] = EYSFTransactDF['AcademicBegin'].astype(int)
EYSFTransactDF['AcademicEnd'] = EYSFTransactDF['AcademicEnd'].astype(int)

#df.dtypes

EYSFTransactDF.loc[(EYSFTransactDF['PSEAS1']>=6)&(EYSFTransactDF['PSEAS2']>=6)&(EYSFTransactDF['PSEAS3']>=6)&
       (EYSFTransactDF['CLLAS1']>=6)&(EYSFTransactDF['CLLAS2']>=6)&(EYSFTransactDF['CLLAS3']>=6)&(EYSFTransactDF['CLLAS4']>=6)&
       (EYSFTransactDF['PSRNAS1']>=6)&(EYSFTransactDF['PSRNAS2']>=6)&(EYSFTransactDF['PSRNAS3']>=6)&
       (EYSFTransactDF['RKUW']>=6)&(EYSFTransactDF['RICD']>=6)&(EYSFTransactDF['RIPD']>=6)&(EYSFTransactDF['EYFSPTotal']>=78), 'newGLD'] = True   
                                     


# In[32]:


EYSFTransactDF.columns


# <div class="alert alert-block alert-info">
#     <h2>
#     1. 47.7% Women students appeared during the academic year 2006-2007 <br/>
#     2. 52.2% Men students appeared during the academic year 2006-2007 <br/>
#     </h2>
# </div>
#     

# <div class="alert alert-block alert-warning">
#     <h3> Due to Statutary limitation of the EYFSP - we will avoid the records from 2002-2005</h3>
# </div>

# In[33]:


statutaryYears2007_2009 = EYSFTransactDF[EYSFTransactDF["AcademicBegin"]>=2006]
statutaryYears2007_2009 = statutaryYears2007_2009.rename(columns={'Persistent_NEET_YN_over_4months':'Persistent_NEET'})
#statutaryYears2007_2009.dtypes
statutaryYears2007_2009


# In[34]:


disp1 = statutaryYears2007_2009[['person_id','PSEAS1','PSEAS2','PSEAS3','CLLAS1','CLLAS2','CLLAS3','CLLAS4','PSRNAS1','PSRNAS2','PSRNAS3','RKUW','RICD','RIPD','PSETotal','CLLTotal','PSRNTotal','EYFSPTotal','newGLD']]
#disp1


# In[35]:


# disp2 = statutaryYears2007_2009[['person_id','newGLD','COMG01','COMG02','COMG03','PHYG04','PHYG05','PSEG06','PSEG07','PSEG08','LITG09','LITG10','MATG11','MATG12','UTWG13','UTWG14','UTWG15','EXPG16','EXPG17']]
# disp2

#disp3 = statutaryYears2007_2009[statutaryYears2007_2009['EYFSPTotal'] >= 78]
disp3 = disp1[disp1['EYFSPTotal'] >= 78]
disp3.newGLD.sum()

#disp3


# <div class="alert alert-block alert-info">
#     <h2>
#     1. 5111 records have EYFSP total greater or equal to 78<br/>
#     2. out of 5111, only 2923 records have GLD attainment <br/>
#     3. These records are interesting to probe further as students have scored abobe the total but have missed on a subject.  further analysis can reveal which subjects students under perform 
#     </h2>
# </div>
#     

# <div class="alert alert-block alert-info">
# <h3> 2923 students have EYFSPTotal >=78 and have Good Level of Development attaintment True </h3>
# <h3> 5108 - 2925 = 2183 have EYFSPTotal >= 78 but Good Level of Development attaintment False </h3>
# # These records are quite interesting for research to see which subjects predicts future NEET #
#     
# </div>

# In[36]:


GraphingData = statutaryYears2007_2009.groupby(['AcademicYear','newGLD','Gender']).agg({
    'Gender':'value_counts',
    'ever_NEET':'sum',
    'Persistent_NEET':'sum',
    #'LSOA_name':'count'
    #'newGLD':'value_counts',
     }).rename(columns={'Gender':'COUNTByGender'})
GraphingData   


# In[37]:


GraphingNonNeetData = statutaryYears2007_2009.groupby(['AcademicYear','newGLD']).agg({
   # 'Gender':'value_counts',
    'ever_NEET':lambda x: (x==False).sum(),
    'Persistent_NEET':lambda x: (x==False).sum(),
    #'LSOA_name':'count'
    #'newGLD':'value_counts',
     }).rename(columns={'Gender':'COUNTByGender'})
GraphingNonNeetData   



# In[38]:


GraphingDataLSOA = statutaryYears2007_2009.groupby(['AcademicYear','newGLD','Gender']).agg({
    'Gender':'value_counts',
    #'ever_NEET':'sum',
    #'Persistent_NEET':'sum',
    'Bradford_YN':'sum'
    #'newGLD':'value_counts',
     }).rename(columns={'Gender':'COUNTByGender'})
GraphingDataLSOA   



# In[39]:


matrix = GraphingData.reset_index()
# plt.bar(X_axis + 0.8, GraphingData['ever_NEET'][2], 0.4, label = 'Female Ever Neet ')
# plt.bar(X_axis + 1.2, GraphingData['ever_NEET'][3], 0.4, label = 'Male Ever Neet')


# In[40]:


GLDTrueGenderData = matrix[(matrix['newGLD']==True)]
GLDFalseGenderData = matrix[(matrix['newGLD']==False)]
GLDTrueGenderData = GLDTrueGenderData.set_index(['AcademicYear','Gender'])
print(GLDTrueGenderData)
GLDFalseGenderData = GLDFalseGenderData.set_index(['AcademicYear','Gender'])
print(GLDFalseGenderData)



# In[41]:


matrixLSOA = GraphingDataLSOA.reset_index()
GLDTrueGenderDataLSOA = matrixLSOA[(matrixLSOA['newGLD']==True)]
GLDFalseGenderDataLSOA = matrixLSOA[(matrixLSOA['newGLD']==False)]
GLDTrueGenderDataLSOA = GLDTrueGenderDataLSOA.set_index(['AcademicYear','Gender'])
print(GLDTrueGenderDataLSOA)
GLDFalseGenderDataLSOA = GLDFalseGenderDataLSOA.set_index(['AcademicYear','Gender'])
print(GLDFalseGenderDataLSOA)

from matplotlib import ticker
import matplotlib.pyplot as plt

GLDFalseGenderDataLSOA.drop(GLDFalseGenderDataLSOA.tail(2).index,
        inplace = True)
lx=GLDFalseGenderDataLSOA.plot(kind='bar')
plt.ylabel('Count of people from Bradford LSOA')

plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
plt.gca().xaxis.set_tick_params(rotation=0)

for bar in lx.patches:
    height = bar.get_height()
    lx.text(bar.get_x() + bar.get_width() / 2.0, height, f' {height:.2f}', fontsize=10,
            ha='center')
    lx.set_title("count of GLD non attainment based out of Bradford LSOA")
plt.tight_layout()
plt.show()


# <div class="alert alert-block alert-info">
#     <h3>
#         <div>
#     2006 - GLD True ->  2914 -Ever NEET ->119 =>4.07%-> Persistent NEET 1.54%  <br/>
#         -     Female-> 20.01% -> Ever NEET -> 2.2%-> Persistent NEET 0.79% <br/>
#         -     Male -> 15.50% -> Ever NEET -> 1.78%-> Persistent NEET 0.76% 
# <br/>
#     2006 - GLD False-> 5290 - Ever NEET ->570 =>10.77%-> Persistent NEET 4.74%  <br/>
#         -     Female-> 27.74% -> Ever NEET -> 4.04% -> Persistent NEET 1.81% <br/>
#         -     Male -> 36.74% -> Ever NEET -> 6.72% -> Persistent NEET 2.92%  
#         </div>
# <br/>
#     2007 - GLD True ->  13 - Ever NEET ->1 => 7%
# <br/>
#     2007 - GLD False-> 39 - Ever NEET ->1 => 2.56%
#     </h3>
# </div>

# In[42]:


from matplotlib import ticker
import matplotlib.pyplot as plt

GLDTrueGenderData.drop(GLDTrueGenderData.tail(2).index,
        inplace = True)
ax=GLDTrueGenderData.plot(kind='bar')
plt.ylabel('GLD Status by Gender')

plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
plt.gca().xaxis.set_tick_params(rotation=0)

for bar in ax.patches:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2.0, height, f' {height:.2f}', fontsize=10,
            ha='center')
    ax.set_title("count of GLD attainment True vs NEET based on Gender")
plt.tight_layout()
plt.show()


# In[43]:


GLDFalseGenderData.drop(GLDFalseGenderData.tail(2).index,
        inplace = True)


bx=GLDFalseGenderData.plot(kind='bar')
plt.ylabel('GLD Status by Gender')
plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
plt.gca().xaxis.set_tick_params(rotation=0)

for bar in bx.patches:
    height = bar.get_height()
    bx.text(bar.get_x() + bar.get_width() / 2.0, height, f' {height:.2f}', fontsize=10,
            ha='center')
    bx.set_title("count of GLD attainment False vs NEET based on Gender")


# In[44]:


corrDF = statutaryYears2007_2009[['newGLD','ever_NEET','Persistent_NEET','PSEAS1','PSEAS2','PSEAS3','CLLAS1','CLLAS2','CLLAS3','CLLAS4','PSRNAS1','PSRNAS2','PSRNAS3','RKUW','RICD','RIPD','PSETotal','CLLTotal','PSRNTotal','EYFSPTotal']]

pearson=corrDF.corr(method='pearson')
pearson


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

# Let's see the correlation matrix 
plt.figure(figsize = (20,10))        # Size of the figure
sns.heatmap(corrDF.corr(method='pearson'),annot = True)
plt.show()


# In[45]:


X=statutaryYears2007_2009[['RIPD','RICD','RKUW','PSETotal','CLLTotal','PSRNTotal','EYFSPTotal']]
y=statutaryYears2007_2009[['newGLD']]
### The data has to be divided in training and test set. 
from sklearn.model_selection import train_test_split 
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.35)


# In[46]:


len(statutaryYears2007_2009)


# In[47]:


GLDTrueData = GLDTrueGenderData.reset_index()
GLDFalseData = GLDFalseGenderData.reset_index()
GLDFalseData


# In[48]:


GLDFailedCount = GLDFalseData.COUNTByGender[0] + GLDFalseData.COUNTByGender[1]
GLDFailedCount


# In[49]:


# Lets find out the number of students who has failed to attaintment condition

test_Fail_EYFSP = statutaryYears2007_2009.groupby(['AcademicYear','newGLD','Gender']).agg({
    'Gender':'count', 
    #'newGLD':lambda x: (x==False).sum(),
    'ever_NEET':'sum',
    'Persistent_NEET':'sum',
    'PSEAS1':lambda ts: (ts < 6).sum(),
    'PSEAS2':lambda ts: (ts < 6).sum(),
    'PSEAS3':lambda ts: (ts < 6).sum(),
   # 'PSETotal': 'sum',
    'CLLAS1':lambda ts: (ts < 6).sum(),
    'CLLAS2':lambda ts: (ts < 6).sum(),
    'CLLAS3':lambda ts: (ts < 6).sum(),
    'CLLAS4':lambda ts: (ts < 6).sum(),
   # 'CLLTotal':'sum',
    'PSRNAS1':lambda ts: (ts < 6).sum(),
    'PSRNAS2':lambda ts: (ts < 6).sum(),
    'PSRNAS3':lambda ts: (ts < 6).sum(),
   # 'PSRNTotal':lambda ts: (ts >= 6).sum(),
    'RKUW':lambda ts: (ts < 6).sum(),
    'RIPD':lambda ts: (ts < 6).sum(),
    'RICD':lambda ts: (ts < 6).sum(),
    'EYFSPTotal':lambda ts: (ts <78 ).sum()
    }).rename(columns={'Gender':'COUNTByGender'})

test_Fail_EYFSP



# In[50]:


# Lets find out the number of students who has failed to attaintment condition

statsCountNEET = statutaryYears2007_2009.groupby(['AcademicYear', 'newGLD']).agg({ 
    #'newGLD':lambda x: (x==False).sum(),
    'ever_NEET':lambda x: (x==True).sum(),
    'Persistent_NEET': lambda x: (x==True).sum(),
    'PSEAS1':lambda ts: (ts < 6).sum(),
    'PSEAS2':lambda ts: (ts < 6).sum(),
    'PSEAS3':lambda ts: (ts < 6).sum(),
   # 'PSETotal': 'sum',
    'CLLAS1':lambda ts: (ts < 6).sum(),
    'CLLAS2':lambda ts: (ts < 6).sum(),
    'CLLAS3':lambda ts: (ts < 6).sum(),
    'CLLAS4':lambda ts: (ts < 6).sum(),
   # 'CLLTotal':'sum',
    'PSRNAS1':lambda ts: (ts < 6).sum(),
    'PSRNAS2':lambda ts: (ts < 6).sum(),
    'PSRNAS3':lambda ts: (ts < 6).sum(),
   # 'PSRNTotal':lambda ts: (ts >= 6).sum(),
    'RKUW':lambda ts: (ts < 6).sum(),
    'RIPD':lambda ts: (ts < 6).sum(),
    'RICD':lambda ts: (ts < 6).sum(),
    'EYFSPTotal':lambda ts: (ts <78 ).sum()
    })
#.rename(columns={'Gender':'COUNTByGender'})

statsCountNEET




# In[51]:


sumval = (test_Fail_EYFSP/5285)*100
sumval


# In[52]:


sumvalALL = (statsCountNEET/5285)*100
sumvalALL


# In[53]:


genderPercentOver = sumval.reset_index()
genderPercentOver.drop(genderPercentOver.tail(4).index,
        inplace = True)
genderPercentOverFemale = genderPercentOver.query("newGLD == False & Gender.str.contains('F')")
genderPercentOverMale = genderPercentOver.query("newGLD == False & Gender.str.contains('M')")

#labels1 = ['PSEAS1','PSEAS2','PSEAS3','CLLAS1','CLLAS2','CLLAS3','CLLAS4','PSRNAS1','PSRNAS2','PSRNAS3','RKUW','RICD','RIPD']
genderPercentOverFemale= genderPercentOverFemale[['ever_NEET','Persistent_NEET','PSEAS1','PSEAS2','PSEAS3','CLLAS1','CLLAS2','CLLAS3','CLLAS4','PSRNAS1','PSRNAS2','PSRNAS3','RKUW','RICD','RIPD']]
genderPercentOverMale= genderPercentOverMale[['ever_NEET','Persistent_NEET','PSEAS1','PSEAS2','PSEAS3','CLLAS1','CLLAS2','CLLAS3','CLLAS4','PSRNAS1','PSRNAS2','PSRNAS3','RKUW','RICD','RIPD']]
genderPercentOverMale


# In[54]:


PercentOver = sumvalALL.reset_index()
PercentOver.drop(PercentOver.tail(2).index,
        inplace = True)
PercentOverGP = PercentOver.query("newGLD == False")
PercentOverGP= PercentOverGP[['newGLD','ever_NEET','Persistent_NEET','PSEAS1','PSEAS2','PSEAS3','CLLAS1','CLLAS2','CLLAS3','CLLAS4','PSRNAS1','PSRNAS2','PSRNAS3','RKUW','RICD','RIPD']]
PercentOverGP = PercentOverGP.rename(columns={'PSEAS1':'Personal Social Emotional - readiness Classroom',
                      'PSEAS2':'Personal Social Emotional - readiness Relationship',
                      'PSEAS3':'Personal Social Emotional - readiness Expressive',
                      'CLLAS1':'Communication, language, literacy - Listening',
                      'CLLAS2':'Communication, language, literacy - Reading',
                      'CLLAS3':'Communication, language, literacy - Reading Books',
                      'CLLAS4':'Communication, language, literacy - Communication',
                      'PSRNAS1':'Problem solving Reasoning and Numeracy - Counting',
                      'PSRNAS2':'Problem solving Reasoning and Numeracy - Recognition',
                      'PSRNAS3':'Problem solving Reasoning and Numeracy - Practicing',
                      'RKUW':'Understanding of the World',
                      'RIPD':'Physical Development',
                      'RICD':'Creative Development'})


# In[55]:


# import libraries
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

tfs = PercentOverGP
tfs1 = pd.melt(tfs, id_vars = "newGLD")

tfs1 = tfs1.rename(columns={"variable": "NEET and Domains"})

g=sns.catplot(x = 'newGLD', y='value',hue = 'NEET and Domains',data=tfs1, kind='bar', width = 1, legend=True, height=6, aspect=2, palette = 'pastel')
ax = g.facet_axis(0, 0)  # or ax = g.axes.flat[0]

# iterate through the axes containers
for c in ax.containers:
    labels = [f'{(v.get_height() ):.1f}' for v in c]
    ax.bar_label(c, labels=labels, label_type='edge')
#sns.despine()
plt.title("GLD Failed percenatge w.r.t NEET and Domains")
plt.xlabel("Percentage failure in Good Level of Development Domain wise")
#plt.xticks(rotation=90)
plt.show()




# In[56]:


from matplotlib import ticker
import matplotlib.pyplot as plt

plt.style.use('ggplot')
fig, (ax1,ax2) = plt.subplots(nrows=1,ncols=2,figsize=(24,12))
plt.subplots_adjust(wspace=0.2)

genderPercentOverFemale.plot(kind='bar',ax=ax1)
ax1.set_ylabel('subject by percentages')

for bar in ax1.patches:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width() / 2.0, height, f' {height:.2f}', fontsize=12,
            ha='center')
    ax1.set_title("Percentage of GLD non attainment by Female")
    

genderPercentOverMale.plot(kind='bar', ax=ax2)
ax2.set_ylabel('subject by percentages')
for bar in ax2.patches:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width() / 2.0, height, f' {height:.2f}', fontsize=12,
            ha='center')
    ax2.set_title("Percentage of GLD non attainment by Male")
    
plt.show() 
 


# In[57]:


#overall = test_Fail_EYFSP.groupby(level=0).transform('sum')
# overall = sumval.groupby(level=0).transform('sum')
# overall


# In[58]:


# Lets find out the number of students who has failed to attaintment condition

test_Fail_EYFSP_all = statutaryYears2007_2009.groupby(['AcademicYear', 'newGLD']).agg({
   # 'newGLD':lambda x: (x==False).sum(),
    'ever_NEET':'sum',
    'Persistent_NEET':'sum',
    'PSEAS1':lambda ts: (ts < 6).sum(),
    'PSEAS2':lambda ts: (ts < 6).sum(),
    'PSEAS3':lambda ts: (ts < 6).sum(),
   # 'PSETotal': 'sum',
    'CLLAS1':lambda ts: (ts < 6).sum(),
    'CLLAS2':lambda ts: (ts < 6).sum(),
    'CLLAS3':lambda ts: (ts < 6).sum(),
    'CLLAS4':lambda ts: (ts < 6).sum(),
   # 'CLLTotal':'sum',
    'PSRNAS1':lambda ts: (ts < 6).sum(),
    'PSRNAS2':lambda ts: (ts < 6).sum(),
    'PSRNAS3':lambda ts: (ts < 6).sum(),
   # 'PSRNTotal':lambda ts: (ts >= 6).sum(),
    'RKUW':lambda ts: (ts < 6).sum(),
    'RIPD':lambda ts: (ts < 6).sum(),
    'RICD':lambda ts: (ts < 6).sum(),
    'EYFSPTotal':lambda ts: (ts <78 ).sum()
    }).rename(columns={'PSEAS1':'Personal Social Emotional - readiness Classroom',
                      'PSEAS2':'Personal Social Emotional - readiness Relationship',
                      'PSEAS3':'Personal Social Emotional - readiness Expressive',
                      'CLLAS1':'Communication, language, literacy - Listening',
                      'CLLAS2':'Communication, language, literacy - Reading',
                      'CLLAS3':'Communication, language, literacy - Reading Books',
                      'CLLAS4':'Communication, language, literacy - Communication',
                      'PSRNAS1':'Problem solving Reasoning and Numeracy - Counting',
                      'PSRNAS2':'Problem solving Reasoning and Numeracy - Recognition',
                      'PSRNAS3':'Problem solving Reasoning and Numeracy - Practicing',
                      'RKUW':'Understanding of the World',
                      'RIPD':'Physical Development',
                      'RICD':'Creative Development'})

test_Fail_EYFSP_all



# In[59]:


sumvalAll = (test_Fail_EYFSP_all/5285)*100

PercentOverAll = sumvalAll.reset_index()
PercentOverAll.drop(PercentOverAll.tail(2).index,inplace = True)
PercentOverAll


# <div class="alert alert-block alert-info">
#         <div>
#     1. CLLAS4 - Communication, Language and Literacy - Roughly 47% failed in this subject <br/>
#             <li> uses phonic knowledge to write simple regular words </li>
#             <li>begins to form captions and simple sentences, sometimes using punctuation</li>
#             <li>communicates meaning through phrases and simple sentences </li><br/>
#     2. CLLAS2 - Communication, Language and Literacy - Roughly 39.3% failed in this subject <br/>
#             <li>uses phonic knowledge to read simple regular words </li>
#             <li>attempts to read more complex words, using phonic knowledge</li>
#             <li>uses knowledge of letters, sounds and words when reading and writing independently </li><br/>       
#     3. PSRNAS2 - Problem Solving, Reasoning and Numeracy - Roughly 38% failed in this subject<br/>
#             <li> in practical activities and discussion, begins to use the vocabulary involved in adding and subtracting </li>
#             <li> uses developing mathematical ideas and methods to solve practical problems</li>
#             <li>uses a range of strategies for addition and subtraction, inclusion some mental recall of number bonds </li><br/>
# 
#   
# </div>

# In[60]:


# Lets find out the number of students who has failed to attaintment condition
test_Fail_all_EYFSP=statutaryYears2007_2009


# In[61]:


test_Fail_all_EYFSP.loc[(test_Fail_all_EYFSP['PSEAS1']<6)&(test_Fail_all_EYFSP['PSEAS2']<6)&(test_Fail_all_EYFSP['PSEAS3']<6)&
       (test_Fail_all_EYFSP['CLLAS1']<6)&(test_Fail_all_EYFSP['CLLAS2']<6)&(test_Fail_all_EYFSP['CLLAS3']<6)&(test_Fail_all_EYFSP['CLLAS4']<6)&
       (test_Fail_all_EYFSP['PSRNAS1']<6)&(test_Fail_all_EYFSP['PSRNAS2']<6)&(test_Fail_all_EYFSP['PSRNAS3']<6)&
       (test_Fail_all_EYFSP['RKUW']<6)&(test_Fail_all_EYFSP['RICD']<6)&(test_Fail_all_EYFSP['RIPD']<6), 'allSubjecttFail'] = True   

test_Fail_all_EYFSP.allSubjecttFail.sum()


# In[62]:


test_Fail_all_EYFSP.loc[(test_Fail_all_EYFSP['PSEAS1']<6)&(test_Fail_all_EYFSP['PSEAS2']>=6)&(test_Fail_all_EYFSP['PSEAS3']>=6)&
       (test_Fail_all_EYFSP['CLLAS1']>=6)&(test_Fail_all_EYFSP['CLLAS2']>=6)&(test_Fail_all_EYFSP['CLLAS3']>=6)&(test_Fail_all_EYFSP['CLLAS4']>=6)&
       (test_Fail_all_EYFSP['PSRNAS1']>=6)&(test_Fail_all_EYFSP['PSRNAS2']>=6)&(test_Fail_all_EYFSP['PSRNAS3']>=6)&
       (test_Fail_all_EYFSP['RKUW']>=6)&(test_Fail_all_EYFSP['RICD']>=6)&(test_Fail_all_EYFSP['RIPD']>=6), 'PSEAS1Fail'] = True   
test_Fail_all_EYFSP.PSEAS1Fail.sum()


# In[63]:


test_Fail_all_EYFSP.loc[(test_Fail_all_EYFSP['PSEAS1']>=6)&(test_Fail_all_EYFSP['PSEAS2']<6)&(test_Fail_all_EYFSP['PSEAS3']>=6)&
       (test_Fail_all_EYFSP['CLLAS1']>=6)&(test_Fail_all_EYFSP['CLLAS2']>=6)&(test_Fail_all_EYFSP['CLLAS3']>=6)&(test_Fail_all_EYFSP['CLLAS4']>=6)&
       (test_Fail_all_EYFSP['PSRNAS1']>=6)&(test_Fail_all_EYFSP['PSRNAS2']>=6)&(test_Fail_all_EYFSP['PSRNAS3']>=6)&
       (test_Fail_all_EYFSP['RKUW']>=6)&(test_Fail_all_EYFSP['RICD']>=6)&(test_Fail_all_EYFSP['RIPD']>=6), 'PSEAS2Fail'] = True   
test_Fail_all_EYFSP.PSEAS2Fail.sum()


# In[64]:


test_Fail_all_EYFSP.loc[(test_Fail_all_EYFSP['PSEAS1']>=6)&(test_Fail_all_EYFSP['PSEAS2']>=6)&(test_Fail_all_EYFSP['PSEAS3']<6)&
       (test_Fail_all_EYFSP['CLLAS1']>=6)&(test_Fail_all_EYFSP['CLLAS2']>=6)&(test_Fail_all_EYFSP['CLLAS3']>=6)&(test_Fail_all_EYFSP['CLLAS4']>=6)&
       (test_Fail_all_EYFSP['PSRNAS1']>=6)&(test_Fail_all_EYFSP['PSRNAS2']>=6)&(test_Fail_all_EYFSP['PSRNAS3']>=6)&
       (test_Fail_all_EYFSP['RKUW']>=6)&(test_Fail_all_EYFSP['RICD']>=6)&(test_Fail_all_EYFSP['RIPD']>=6), 'PSEAS3Fail'] = True   
test_Fail_all_EYFSP.PSEAS3Fail.sum()


# In[65]:


test_Fail_all_EYFSP.loc[(test_Fail_all_EYFSP['PSEAS1']>=6)&(test_Fail_all_EYFSP['PSEAS2']>=6)&(test_Fail_all_EYFSP['PSEAS3']>=6)&
       (test_Fail_all_EYFSP['CLLAS1']<6)&(test_Fail_all_EYFSP['CLLAS2']>=6)&(test_Fail_all_EYFSP['CLLAS3']>=6)&(test_Fail_all_EYFSP['CLLAS4']>=6)&
       (test_Fail_all_EYFSP['PSRNAS1']>=6)&(test_Fail_all_EYFSP['PSRNAS2']>=6)&(test_Fail_all_EYFSP['PSRNAS3']>=6)&
       (test_Fail_all_EYFSP['RKUW']>=6)&(test_Fail_all_EYFSP['RICD']>=6)&(test_Fail_all_EYFSP['RIPD']>=6), 'CLLAS1Fail'] = True   
test_Fail_all_EYFSP.CLLAS1Fail.sum()


# In[66]:


test_Fail_all_EYFSP.loc[(test_Fail_all_EYFSP['PSEAS1']>=6)&(test_Fail_all_EYFSP['PSEAS2']>=6)&(test_Fail_all_EYFSP['PSEAS3']>=6)&
       (test_Fail_all_EYFSP['CLLAS1']>=6)&(test_Fail_all_EYFSP['CLLAS2']<6)&(test_Fail_all_EYFSP['CLLAS3']>=6)&(test_Fail_all_EYFSP['CLLAS4']>=6)&
       (test_Fail_all_EYFSP['PSRNAS1']>=6)&(test_Fail_all_EYFSP['PSRNAS2']>=6)&(test_Fail_all_EYFSP['PSRNAS3']>=6)&
       (test_Fail_all_EYFSP['RKUW']>=6)&(test_Fail_all_EYFSP['RICD']>=6)&(test_Fail_all_EYFSP['RIPD']>=6), 'CLLAS2Fail'] = True   
test_Fail_all_EYFSP.CLLAS2Fail.sum()


# In[67]:


test_Fail_all_EYFSP.loc[(test_Fail_all_EYFSP['PSEAS1']>=6)&(test_Fail_all_EYFSP['PSEAS2']>=6)&(test_Fail_all_EYFSP['PSEAS3']>=6)&
       (test_Fail_all_EYFSP['CLLAS1']>=6)&(test_Fail_all_EYFSP['CLLAS2']>=6)&(test_Fail_all_EYFSP['CLLAS3']<6)&(test_Fail_all_EYFSP['CLLAS4']>=6)&
       (test_Fail_all_EYFSP['PSRNAS1']>=6)&(test_Fail_all_EYFSP['PSRNAS2']>=6)&(test_Fail_all_EYFSP['PSRNAS3']>=6)&
       (test_Fail_all_EYFSP['RKUW']>=6)&(test_Fail_all_EYFSP['RICD']>=6)&(test_Fail_all_EYFSP['RIPD']>=6), 'CLLAS3Fail'] = True   
test_Fail_all_EYFSP.CLLAS3Fail.sum()


# In[68]:


test_Fail_all_EYFSP.loc[(test_Fail_all_EYFSP['PSEAS1']>=6)&(test_Fail_all_EYFSP['PSEAS2']>=6)&(test_Fail_all_EYFSP['PSEAS3']>=6)&
       (test_Fail_all_EYFSP['CLLAS1']>=6)&(test_Fail_all_EYFSP['CLLAS2']>=6)&(test_Fail_all_EYFSP['CLLAS3']>=6)&(test_Fail_all_EYFSP['CLLAS4']<6)&
       (test_Fail_all_EYFSP['PSRNAS1']>=6)&(test_Fail_all_EYFSP['PSRNAS2']>=6)&(test_Fail_all_EYFSP['PSRNAS3']>=6)&
       (test_Fail_all_EYFSP['RKUW']>=6)&(test_Fail_all_EYFSP['RICD']>=6)&(test_Fail_all_EYFSP['RIPD']>=6), 'CLLAS4Fail'] = True   
test_Fail_all_EYFSP.CLLAS4Fail.sum()


# In[69]:


test_Fail_all_EYFSP.loc[(test_Fail_all_EYFSP['PSEAS1']>=6)&(test_Fail_all_EYFSP['PSEAS2']>=6)&(test_Fail_all_EYFSP['PSEAS3']>=6)&
       (test_Fail_all_EYFSP['CLLAS1']>=6)&(test_Fail_all_EYFSP['CLLAS2']>=6)&(test_Fail_all_EYFSP['CLLAS3']>=6)&(test_Fail_all_EYFSP['CLLAS4']>=6)&
       (test_Fail_all_EYFSP['PSRNAS1']<6)&(test_Fail_all_EYFSP['PSRNAS2']>=6)&(test_Fail_all_EYFSP['PSRNAS3']>=6)&
       (test_Fail_all_EYFSP['RKUW']>=6)&(test_Fail_all_EYFSP['RICD']>=6)&(test_Fail_all_EYFSP['RIPD']>=6), 'PSRNAS1Fail'] = True   
test_Fail_all_EYFSP.PSRNAS1Fail.sum()


# In[70]:


test_Fail_all_EYFSP.loc[(test_Fail_all_EYFSP['PSEAS1']>=6)&(test_Fail_all_EYFSP['PSEAS2']>=6)&(test_Fail_all_EYFSP['PSEAS3']>=6)&
       (test_Fail_all_EYFSP['CLLAS1']>=6)&(test_Fail_all_EYFSP['CLLAS2']>=6)&(test_Fail_all_EYFSP['CLLAS3']>=6)&(test_Fail_all_EYFSP['CLLAS4']>=6)&
       (test_Fail_all_EYFSP['PSRNAS1']>=6)&(test_Fail_all_EYFSP['PSRNAS2']<6)&(test_Fail_all_EYFSP['PSRNAS3']>=6)&
       (test_Fail_all_EYFSP['RKUW']>=6)&(test_Fail_all_EYFSP['RICD']>=6)&(test_Fail_all_EYFSP['RIPD']>=6), 'PSRNAS2Fail'] = True   
test_Fail_all_EYFSP.PSRNAS2Fail.sum()


# In[71]:


test_Fail_all_EYFSP.loc[(test_Fail_all_EYFSP['PSEAS1']>=6)&(test_Fail_all_EYFSP['PSEAS2']>=6)&(test_Fail_all_EYFSP['PSEAS3']>=6)&
       (test_Fail_all_EYFSP['CLLAS1']>=6)&(test_Fail_all_EYFSP['CLLAS2']>=6)&(test_Fail_all_EYFSP['CLLAS3']>=6)&(test_Fail_all_EYFSP['CLLAS4']>=6)&
       (test_Fail_all_EYFSP['PSRNAS1']>=6)&(test_Fail_all_EYFSP['PSRNAS2']>=6)&(test_Fail_all_EYFSP['PSRNAS3']<6)&
       (test_Fail_all_EYFSP['RKUW']>=6)&(test_Fail_all_EYFSP['RICD']>=6)&(test_Fail_all_EYFSP['RIPD']>=6), 'PSRNAS3Fail'] = True   
test_Fail_all_EYFSP.PSRNAS3Fail.sum()


# In[72]:


test_Fail_all_EYFSP.loc[(test_Fail_all_EYFSP['PSEAS1']>=6)&(test_Fail_all_EYFSP['PSEAS2']>=6)&(test_Fail_all_EYFSP['PSEAS3']>=6)&
       (test_Fail_all_EYFSP['CLLAS1']>=6)&(test_Fail_all_EYFSP['CLLAS2']>=6)&(test_Fail_all_EYFSP['CLLAS3']>=6)&(test_Fail_all_EYFSP['CLLAS4']>=6)&
       (test_Fail_all_EYFSP['PSRNAS1']>=6)&(test_Fail_all_EYFSP['PSRNAS2']>=6)&(test_Fail_all_EYFSP['PSRNAS3']>=6)&
       (test_Fail_all_EYFSP['RKUW']<6)&(test_Fail_all_EYFSP['RICD']>=6)&(test_Fail_all_EYFSP['RIPD']>=6), 'RKUWFail'] = True   
test_Fail_all_EYFSP.RKUWFail.sum()


# In[73]:


test_Fail_all_EYFSP.loc[(test_Fail_all_EYFSP['PSEAS1']>=6)&(test_Fail_all_EYFSP['PSEAS2']>=6)&(test_Fail_all_EYFSP['PSEAS3']>=6)&
       (test_Fail_all_EYFSP['CLLAS1']>=6)&(test_Fail_all_EYFSP['CLLAS2']>=6)&(test_Fail_all_EYFSP['CLLAS3']>=6)&(test_Fail_all_EYFSP['CLLAS4']>=6)&
       (test_Fail_all_EYFSP['PSRNAS1']>=6)&(test_Fail_all_EYFSP['PSRNAS2']>=6)&(test_Fail_all_EYFSP['PSRNAS3']>=6)&
       (test_Fail_all_EYFSP['RKUW']>=6)&(test_Fail_all_EYFSP['RICD']<6)&(test_Fail_all_EYFSP['RIPD']>=6), 'RICDFail'] = True   
test_Fail_all_EYFSP.RICDFail.sum()


# In[74]:


test_Fail_all_EYFSP.loc[(test_Fail_all_EYFSP['PSEAS1']>=6)&(test_Fail_all_EYFSP['PSEAS2']>=6)&(test_Fail_all_EYFSP['PSEAS3']>=6)&
       (test_Fail_all_EYFSP['CLLAS1']>=6)&(test_Fail_all_EYFSP['CLLAS2']>=6)&(test_Fail_all_EYFSP['CLLAS3']>=6)&(test_Fail_all_EYFSP['CLLAS4']>=6)&
       (test_Fail_all_EYFSP['PSRNAS1']>=6)&(test_Fail_all_EYFSP['PSRNAS2']>=6)&(test_Fail_all_EYFSP['PSRNAS3']>=6)&
       (test_Fail_all_EYFSP['RKUW']>=6)&(test_Fail_all_EYFSP['RICD']>=6)&(test_Fail_all_EYFSP['RIPD']<6), 'RIPDFail'] = True   
test_Fail_all_EYFSP.RIPDFail.sum()


# In[75]:


test_Fail_all_EYFSP.loc[(test_Fail_all_EYFSP['PSEAS1']>=6)&(test_Fail_all_EYFSP['PSEAS2']>=6)&(test_Fail_all_EYFSP['PSEAS3']>=6)&
       (test_Fail_all_EYFSP['CLLAS1']>=6)&(test_Fail_all_EYFSP['CLLAS2']>=6)&(test_Fail_all_EYFSP['CLLAS3']>=6)&(test_Fail_all_EYFSP['CLLAS4']>=6)&
       (test_Fail_all_EYFSP['PSRNAS1']>=6)&(test_Fail_all_EYFSP['PSRNAS2']>=6)&(test_Fail_all_EYFSP['PSRNAS3']>=6)&
       (test_Fail_all_EYFSP['RKUW']>=6)&(test_Fail_all_EYFSP['RICD']>=6)&(test_Fail_all_EYFSP['RIPD']>=6)&(test_Fail_all_EYFSP['EYFSPTotal']<78), 'EYFSPTOTFail'] = True   
test_Fail_all_EYFSP.EYFSPTOTFail.sum()


# In[76]:


test_Fail_all_EYFSP.loc[(test_Fail_all_EYFSP['EYFSPTotal']<78), 'EYFSPTOTFail'] = True   
test_Fail_all_EYFSP.EYFSPTOTFail.sum()


# In[77]:


test_Fail_all_EYFSP['allSubjecttFail'] = test_Fail_all_EYFSP['allSubjecttFail'].apply(lambda set_: False if pd.isna(set_)== True else set_)
test_Fail_all_EYFSP['PSEAS1Fail'] = test_Fail_all_EYFSP['PSEAS1Fail'].apply(lambda set_: False if pd.isna(set_)== True else set_)
test_Fail_all_EYFSP['PSEAS2Fail'] = test_Fail_all_EYFSP['PSEAS2Fail'].apply(lambda set_: False if pd.isna(set_)== True else set_)
test_Fail_all_EYFSP['PSEAS3Fail'] = test_Fail_all_EYFSP['PSEAS3Fail'].apply(lambda set_: False if pd.isna(set_)== True else set_)

test_Fail_all_EYFSP['CLLAS1Fail'] = test_Fail_all_EYFSP['CLLAS1Fail'].apply(lambda set_: False if pd.isna(set_)== True else set_)
test_Fail_all_EYFSP['CLLAS2Fail'] = test_Fail_all_EYFSP['CLLAS2Fail'].apply(lambda set_: False if pd.isna(set_)== True else set_)
test_Fail_all_EYFSP['CLLAS3Fail'] = test_Fail_all_EYFSP['CLLAS3Fail'].apply(lambda set_: False if pd.isna(set_)== True else set_)
test_Fail_all_EYFSP['CLLAS4Fail'] = test_Fail_all_EYFSP['CLLAS4Fail'].apply(lambda set_: False if pd.isna(set_)== True else set_)

test_Fail_all_EYFSP['PSRNAS1Fail'] = test_Fail_all_EYFSP['PSRNAS1Fail'].apply(lambda set_: False if pd.isna(set_)== True else set_)
test_Fail_all_EYFSP['PSRNAS2Fail'] = test_Fail_all_EYFSP['PSRNAS2Fail'].apply(lambda set_: False if pd.isna(set_)== True else set_)
test_Fail_all_EYFSP['PSRNAS3Fail'] = test_Fail_all_EYFSP['PSRNAS3Fail'].apply(lambda set_: False if pd.isna(set_)== True else set_)

test_Fail_all_EYFSP['RKUWFail'] = test_Fail_all_EYFSP['RKUWFail'].apply(lambda set_: False if pd.isna(set_)== True else set_)
test_Fail_all_EYFSP['RIPDFail'] = test_Fail_all_EYFSP['RIPDFail'].apply(lambda set_: False if pd.isna(set_)== True else set_)
test_Fail_all_EYFSP['RICDFail'] = test_Fail_all_EYFSP['RICDFail'].apply(lambda set_: False if pd.isna(set_)== True else set_)
test_Fail_all_EYFSP['EYFSPTOTFail'] = test_Fail_all_EYFSP['EYFSPTOTFail'].apply(lambda set_: False if pd.isna(set_)== True else set_)



# In[78]:


test_Fail_all_EYFSPY1 = test_Fail_all_EYFSP.query('AcademicYear == "2006/2007"')

test_Fail_all_EYFSPY1


# In[79]:


count_All = test_Fail_all_EYFSPY1.groupby(['ever_NEET', 'allSubjecttFail']).size()
print(count_All)
count_PSEAS1 = test_Fail_all_EYFSPY1.groupby(['ever_NEET', 'PSEAS1Fail']).size()
count_PSEAS1


# In[80]:


X = test_Fail_all_EYFSPY1.groupby(['AcademicYear','newGLD']).agg({
    'newGLD':'count',
    'allSubjecttFail':'sum', 
    'ever_NEET':lambda x: (x==False).sum(),
    'Persistent_NEET':lambda x: (x==False).sum()}).rename(columns={'newGLD':'count'})
print(X)


# In[81]:


checkForNEETAllDomain = test_Fail_all_EYFSPY1.groupby(['AcademicYear','allSubjecttFail']).agg({
    'newGLD':'count', 
    'ever_NEET':'sum',
    'Persistent_NEET':'sum'}).rename(columns={'newGLD':'count'})
#print(checkForNEETAllDomain)
DomainAll = checkForNEETAllDomain.reset_index()
#failAll = failAll.drop(['allSubjecttFail'], axis=1)
DomainAll['Category'] = "All Domains"
DomainAll = DomainAll[DomainAll['allSubjecttFail'] == True].drop(['allSubjecttFail'], axis=1)
DomainAll = DomainAll.rename(columns={"count": "Failed"})
print(DomainAll)


# In[82]:


checkForNonNEETAllDomain = test_Fail_all_EYFSPY1.groupby(['AcademicYear','allSubjecttFail']).agg({
    'newGLD':'count', 
    'ever_NEET':lambda x: (x==False).sum()}).rename(columns={'newGLD':'count',
                                                                  'ever_NEET':'NON NEET'})
#print(checkForNonNEETAllDomain)
NonNEETALL = checkForNonNEETAllDomain.reset_index()
#failAll = failAll.drop(['allSubjecttFail'], axis=1)
NonNEETALL['Category'] = "All Domains"
NonNEETALL = NonNEETALL[NonNEETALL['allSubjecttFail'] == True].drop(['allSubjecttFail'], axis=1)
NonNEETALL = NonNEETALL.rename(columns={"count": "Failed"})
print(NonNEETALL)


# In[83]:


checkForNEETCLLAS4Fail = test_Fail_all_EYFSPY1.groupby(['AcademicYear','CLLAS4Fail']).agg({
    'CLLAS4Fail':'sum',
    'ever_NEET':'sum',
    'Persistent_NEET':'sum'}).rename(columns={'CLLAS4Fail':'Failed'}) 
checkForNEETCLLAS4Fail
temp1 = checkForNEETCLLAS4Fail.reset_index()
temp1 = temp1[temp1['CLLAS4Fail'] == True]
temp1 = temp1.drop(['CLLAS4Fail'], axis=1)
temp1['Category'] = "Written Comm"
temp1


# In[84]:


nonNEETCLLAS4Fail = test_Fail_all_EYFSPY1.groupby(['AcademicYear','CLLAS4Fail']).agg({
    'CLLAS4Fail':'sum',
    'ever_NEET':lambda x: (x==False).sum()}).rename(columns={'CLLAS4Fail':'Failed',
                                                                  'ever_NEET':'NON NEET'})
nonNEETCLLAS4Fail
nonNEETtemp1 = nonNEETCLLAS4Fail.reset_index()
nonNEETtemp1 = nonNEETtemp1[nonNEETtemp1['CLLAS4Fail'] == True]
nonNEETtemp1 = nonNEETtemp1.drop(['CLLAS4Fail'], axis=1)
nonNEETtemp1['Category'] = "Written Comm"
nonNEETtemp1


# In[85]:


checkForNEETCLLAS2Fail = test_Fail_all_EYFSPY1.groupby(['AcademicYear','CLLAS2Fail']).agg({
    #'newGLD':lambda x: (x==False).sum(),
    'CLLAS2Fail':'sum',
    'ever_NEET':'sum',
    'Persistent_NEET':'sum'}).rename(columns={'CLLAS2Fail':'Failed' 
                                              # ,'ever_NEET':'ever_NEET_Listening_comm',
                                              # 'Persistent_NEET':'Per_NEET_Listening_comm'
                                             })
checkForNEETCLLAS2Fail
temp2 = checkForNEETCLLAS2Fail.reset_index()
temp2 = temp2[temp2['CLLAS2Fail'] == True]
temp2 = temp2.drop(['CLLAS2Fail'], axis=1)
temp2['Category'] = "Listening"
temp2


# In[86]:


nonNEETCLLAS2Fail = test_Fail_all_EYFSPY1.groupby(['AcademicYear','CLLAS2Fail']).agg({
    'CLLAS2Fail':'sum',
    'ever_NEET':lambda x: (x==False).sum()}).rename(columns={'CLLAS2Fail':'Failed',
                                                                  'ever_NEET':'NON NEET'})
nonNEETtemp2 = nonNEETCLLAS2Fail.reset_index()
nonNEETtemp2 = nonNEETtemp2[nonNEETtemp2['CLLAS2Fail'] == True]
nonNEETtemp2 = nonNEETtemp2.drop(['CLLAS2Fail'], axis=1)
nonNEETtemp2['Category'] = "Listening"
nonNEETtemp2


# In[87]:


checkForNEETPSRNAS2Fail = test_Fail_all_EYFSPY1.groupby(['AcademicYear','PSRNAS2Fail']).agg({
    #'newGLD':lambda x: (x==False).sum(),
    'PSRNAS2Fail':'sum',
    'ever_NEET':'sum',
    'Persistent_NEET':'sum'}).rename(columns={'PSRNAS2Fail':'Failed' 
                                       #       ,'ever_NEET':'ever_NEET_counting_M',
                                       #       'Persistent_NEET':'Per_NEET_counting_M'
                                             })
checkForNEETPSRNAS2Fail
temp3 = checkForNEETPSRNAS2Fail.reset_index()
temp3 = temp3[temp3['PSRNAS2Fail'] == True]
temp3 = temp3.drop(['PSRNAS2Fail'], axis=1)
temp3['Category'] = "Measurements"
temp3


# In[88]:


nonNEETPSRNAS2Fail  = test_Fail_all_EYFSPY1.groupby(['AcademicYear','PSRNAS2Fail']).agg({
    'PSRNAS2Fail':'sum',
    'ever_NEET':lambda x: (x==False).sum()}).rename(columns={'PSRNAS2Fail':'Failed',
                                                                  'ever_NEET':'NON NEET'})
nonNEETtemp3 = nonNEETPSRNAS2Fail.reset_index()
nonNEETtemp3 = nonNEETtemp3[nonNEETtemp3['PSRNAS2Fail'] == True]
nonNEETtemp3 = nonNEETtemp3.drop(['PSRNAS2Fail'], axis=1)
nonNEETtemp3['Category'] = "Measurements"
nonNEETtemp3


# In[89]:


checkForNEETCLLAS3Fail = test_Fail_all_EYFSPY1.groupby(['AcademicYear','CLLAS3Fail']).agg({
    #'newGLD':lambda x: (x==False).sum(),
    'CLLAS3Fail':'sum',
    'ever_NEET':'sum',
    'Persistent_NEET':'sum'}).rename(columns={'CLLAS3Fail':'Failed' 
                                              #,'ever_NEET':'ever_NEET_Reading_SB',
                                              #'Persistent_NEET':'Per_NEET_Reading_SB'
                                             })
checkForNEETCLLAS3Fail
temp4 = checkForNEETCLLAS3Fail.reset_index()
temp4 = temp4[temp4['CLLAS3Fail'] == True]
temp4 = temp4.drop(['CLLAS3Fail'], axis=1)
temp4['Category'] = "Reading SB"
temp4


# In[90]:


nonNEETCLLAS3Fail  = test_Fail_all_EYFSPY1.groupby(['AcademicYear','CLLAS3Fail']).agg({
    'CLLAS3Fail':'sum',
    'ever_NEET':lambda x: (x==False).sum()}).rename(columns={'CLLAS3Fail':'Failed',
                                                                  'ever_NEET':'NON NEET'})
nonNEETtemp4 = nonNEETCLLAS3Fail.reset_index()
nonNEETtemp4 = nonNEETtemp4[nonNEETtemp4['CLLAS3Fail'] == True]
nonNEETtemp4 = nonNEETtemp4.drop(['CLLAS3Fail'], axis=1)
nonNEETtemp4['Category'] = "Reading SB"
nonNEETtemp4


# In[91]:


checkForNEETPSEAS3Fail = test_Fail_all_EYFSPY1.groupby(['AcademicYear','PSEAS3Fail']).agg({
    #'newGLD':lambda x: (x==False).sum(),
    'PSEAS3Fail':'sum',
    'ever_NEET':'sum',
    'Persistent_NEET':'sum'}).rename(columns={'PSEAS3Fail':'Failed' 
                                              #,'ever_NEET':'ever_NEET_Reading_SB',
                                              #'Persistent_NEET':'Per_NEET_Reading_SB'
                                             })
checkForNEETPSEAS3Fail
temp5 = checkForNEETPSEAS3Fail.reset_index()
temp5 = temp5[temp5['PSEAS3Fail'] == True]
temp5 = temp5.drop(['PSEAS3Fail'], axis=1)
temp5['Category'] = "Emotional Dpt"
temp5


# In[92]:


nonNEETPSEAS3Fail  = test_Fail_all_EYFSPY1.groupby(['AcademicYear','PSEAS3Fail']).agg({
    'PSEAS3Fail':'sum',
    'ever_NEET':lambda x: (x==False).sum()}).rename(columns={'PSEAS3Fail':'Failed',
                                                                  'ever_NEET':'NON NEET'})
nonNEETtemp5 = nonNEETPSEAS3Fail.reset_index()
nonNEETtemp5 = nonNEETtemp5[nonNEETtemp5['PSEAS3Fail'] == True]
nonNEETtemp5 = nonNEETtemp5.drop(['PSEAS3Fail'], axis=1)
nonNEETtemp5['Category'] = "Emotional Dpt"
nonNEETtemp5


# In[93]:


checkForNEETRKUWFail = test_Fail_all_EYFSPY1.groupby(['AcademicYear','RKUWFail']).agg({
    #'newGLD':lambda x: (x==False).sum(),
    'RKUWFail':'sum',
    'ever_NEET':'sum',
    'Persistent_NEET':'sum'}).rename(columns={'RKUWFail':'Failed' 
                                              #,'ever_NEET':'ever_NEET_Reading_SB',
                                              #'Persistent_NEET':'Per_NEET_Reading_SB'
                                             })
checkForNEETRKUWFail
temp6 = checkForNEETRKUWFail.reset_index()
temp6 = temp6[temp6['RKUWFail'] == True]
temp6 = temp6.drop(['RKUWFail'], axis=1)
temp6['Category'] = "Understanding the World"
temp6


# In[94]:


nonNEETRKUWFail  = test_Fail_all_EYFSPY1.groupby(['AcademicYear','RKUWFail']).agg({
    'RKUWFail':'sum',
    'ever_NEET':lambda x: (x==False).sum()}).rename(columns={'RKUWFail':'Failed',
                                                                  'ever_NEET':'NON NEET'})
nonNEETtemp6 = nonNEETRKUWFail.reset_index()
nonNEETtemp6 = nonNEETtemp6[nonNEETtemp6['RKUWFail'] == True]
nonNEETtemp6 = nonNEETtemp6.drop(['RKUWFail'], axis=1)
nonNEETtemp6['Category'] = "Emotional Dpt"
nonNEETtemp6


# In[95]:


FailAllStats = pd.concat([DomainAll,temp1,temp3,temp2,temp4,temp5,temp6])
FailAllStats


# In[96]:


FailAllNONNEETStats = pd.concat([NonNEETALL,nonNEETtemp1,nonNEETtemp3,nonNEETtemp2,nonNEETtemp4,nonNEETtemp5,nonNEETtemp6])
FailAllNONNEETStats


# In[97]:


rfs = FailAllStats.drop(['AcademicYear'],axis=1)
rfs1 = pd.melt(rfs, id_vars = "Category")
#dfs1
rfs1 = rfs1.rename(columns={"variable": "Failed Count"})

# #print(dfs1)
# fig, ax = plt.subplots(figsize=(17, 10))
# fig.patch.set_visible(False)

g=sns.catplot(x = 'Category', y='value',hue = 'Failed Count',data=rfs1, kind='bar', width = 1, legend=True, height=6, aspect=2, palette = 'pastel')
ax = g.facet_axis(0, 0)  # or ax = g.axes.flat[0]

# iterate through the axes containers
for c in ax.containers:
    labels = [f'{(v.get_height() ):.1f}' for v in c]
    ax.bar_label(c, labels=labels, label_type='edge')
#sns.despine()
plt.title("Top 5 Domain wise Failure NEET count")
plt.xlabel("Count of Failure Domain wise and NEET Count per Domain Failure")
#plt.xticks(rotation=90)
plt.show()




tfs = FailAllNONNEETStats.drop(['AcademicYear'],axis=1)
tfs1 = pd.melt(tfs, id_vars = "Category")
#dfs1
tfs1 = tfs1.rename(columns={"variable": "Failed Count"})

g=sns.catplot(x = 'Category', y='value',hue = 'Failed Count',data=tfs1, kind='bar', width = 1, legend=True, height=6, aspect=2)
ax = g.facet_axis(0, 0)  # or ax = g.axes.flat[0]

# iterate through the axes containers
for c in ax.containers:
    labels = [f'{(v.get_height() ):.1f}' for v in c]
    ax.bar_label(c, labels=labels, label_type='edge')
#sns.despine()
plt.title("Top 5 Domain wise Failure NON NEET count")
plt.xlabel("Count of Failure Domain wise and NON NEET Count per Domain Failure")
#plt.xticks(rotation=90)
plt.show()





# In[98]:


graphFail =  test_Fail_all_EYFSPY1.groupby(['AcademicYear','Gender']).agg({
    'Gender':'count', 
    'newGLD':lambda x: (x==False).sum(),
    'ever_NEET':'sum',
    'Persistent_NEET':'sum',
    'allSubjecttFail':'sum',
    'PSEAS1Fail':'sum',
    'PSEAS2Fail':'sum',
    'PSEAS3Fail':'sum',
    'CLLAS1Fail':'sum',
    'CLLAS2Fail':'sum',
    'CLLAS3Fail':'sum',
    'CLLAS4Fail':'sum',
    'PSRNAS1Fail':'sum',
    'PSRNAS2Fail':'sum',
    'PSRNAS3Fail':'sum',
    'RKUWFail':'sum',
    'RICDFail':'sum',
    'RIPDFail':'sum'
    }).rename(columns={'Gender':'ByGender'})
graphFail


# In[99]:


sumvalFail = (graphFail/8205)*100
sumvalFail   


# In[100]:


# subjectFailOver = sumvalFail.reset_index()
# subjectFailOver.drop(subjectFailOver.tail(2).index,
#         inplace = True)
# subjectFailOverFemale = subjectFailOver.query("Gender.str.contains('F')")
# subjectFailOverMale = subjectFailOver.query("Gender.str.contains('M')")

# subjectFailOverFemale= subjectFailOverFemale[['allSubjecttFail','PSEAS1Fail','PSEAS2Fail','PSEAS3Fail','CLLAS1Fail','CLLAS2Fail','CLLAS3Fail','CLLAS4Fail','PSRNAS1Fail','PSRNAS2Fail','PSRNAS3Fail','RKUWFail','RICDFail','RIPDFail']]
# subjectFailOverMale= subjectFailOverMale[['allSubjecttFail','PSEAS1Fail','PSEAS2Fail','PSEAS3Fail','CLLAS1Fail','CLLAS2Fail','CLLAS3Fail','CLLAS4Fail','PSRNAS1Fail','PSRNAS2Fail','PSRNAS3Fail','RKUWFail','RICDFail','RIPDFail']]

# from matplotlib import ticker
# import matplotlib.pyplot as plt

# plt.style.use('ggplot')
# fig, (ax1,ax2) = plt.subplots(nrows=1,ncols=2,figsize=(24,12))
# plt.subplots_adjust(wspace=0.2)

# subjectFailOverFemale.plot(kind='bar',ax=ax1)
# ax1.set_ylabel('Count of people failing each subjects')

# for bar in ax1.patches:
#     height = bar.get_height()
#     ax1.text(bar.get_x() + bar.get_width() / 2.0, height, f' {height:.2f}', fontsize=10,
#             ha='center')
#     ax1.set_title("Count of Female students non attainment of GLD by subjects")
    

# subjectFailOverMale.plot(kind='bar', ax=ax2)
# ax2.set_ylabel('Count of people failing each subjects')
# for bar in ax2.patches:
#     height = bar.get_height()
#     ax2.text(bar.get_x() + bar.get_width() / 2.0, height, f' {height:.2f}', fontsize=10,
#             ha='center')
#     ax2.set_title("Count of Male students non attainment of GLD by subjects")
    
# plt.show() 


# <div class="alert alert-block alert-info">
# 
# <h1> Predicting GLD to NEET Status </h1>
# <h3> Performing Generalised Linear Regression </h3>
# 
# 
# <h3>    Adding other covariates to understand the Statistically significant values</h3>
# </div>

# In[101]:


import pandas as pd
import statsmodels.api as sm
import scipy.stats as stats
modelDF = statutaryYears2007_2009

ct=pd.crosstab(modelDF.ever_NEET,modelDF.newGLD)
oddsratio, pvalue = stats.fisher_exact(ct)
print("odds Ratio, pValue",np.asarray((oddsratio, pvalue)))

# modelDF['newGLD'] = modelDF['newGLD'].apply(lambda set_: 1 if (set_)== True else 0)
modelDF['ever_NEET'] = modelDF['ever_NEET'].apply(lambda set_: 1 if (set_)== True else 0)

# Splitting the data into 2 parts
# train, test = np.split(
#     modelDF.sample(frac=1, random_state=42), [int(0.8 * len(modelDF))]
# )

print(modelDF.ever_NEET.sum())
print(len(modelDF.ever_NEET))


model = sm.formula.glm("ever_NEET ~ C(newGLD, Treatment(reference=False))",
                       family=sm.families.Binomial(), data=modelDF).fit()

print(model.summary())

predictions=model.predict()
print(type(predictions))
print(np.unique(predictions))

unique, counts = np.unique(predictions, return_counts=True)

print(np.asarray((unique, counts)).T)

print(np.unique(modelDF['ever_NEET']))

uniqueT, countsT = np.unique(modelDF['ever_NEET'], return_counts=True)

print(np.asarray((uniqueT, countsT)).T)

predictions_nominal = [1 if x < 0.1 else 0 for x in predictions]
      
print(model.params.Intercept)
odds_ratio=np.exp(model.params.Intercept)
print('odds ratio is ',odds_ratio)


# In[102]:


from sklearn.metrics import confusion_matrix, classification_report
cnf_matrix = confusion_matrix(modelDF["ever_NEET"], 
                       predictions_nominal)
print(cnf_matrix)

print(classification_report(modelDF["ever_NEET"], 
                            predictions_nominal, 
                            digits = 3))

class_names=["0","1"] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label NEET')
plt.xlabel('Predicted label NEET')


# In[103]:


modelDF = statutaryYears2007_2009


print(modelDF.ever_NEET.sum())
print(len(modelDF.ever_NEET))

model_covariates = sm.formula.glm("ever_NEET ~ C(newGLD+CLLAS1, Treatment(reference=True))",
                       family=sm.families.Binomial(), data=modelDF).fit()

print(model_covariates.summary())
predictionsCovarites=model_covariates.predict()

print(np.unique(predictionsCovarites))

unique, counts = np.unique(predictionsCovarites, return_counts=True)

print(np.asarray((unique, counts)).T)

uniqueT, countsT = np.unique(modelDF['ever_NEET'], return_counts=True)

print(np.asarray((uniqueT, countsT)).T)

predictions_Covarites = [1 if x < 0.069 else 0 for x in predictionsCovarites]
print(model_covariates.params.Intercept)
odds_ratio_covariates=np.exp(model_covariates.params.Intercept)
print('odds ratio is ',odds_ratio_covariates)


# In[104]:


cnf_matrix_Covarites = confusion_matrix(modelDF["ever_NEET"], 
                       predictions_Covarites)
print(cnf_matrix_Covarites)

print(classification_report(modelDF["ever_NEET"], 
                            predictions_Covarites, 
                            digits = 3))

class_names=["0","1"] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix_Covarites), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label NEET')
plt.xlabel('Predicted label NEET')


# <div class="alert alert-block alert-info">
# 
# <h3> To understand the predictive power of becoming NEET from other covariates like free school meal disability flag, SEND flag</h3>
#     
# </div>

# In[105]:


personUniqueID = statutaryYears2007_2009["person_id"].unique()
ct=len(personUniqueID)
ct
#8240 unique ids
#pID = map(str, personUniqueID) 
Pid = tuple(personUniqueID)

# sqlWideFormat1 = """ SELECT person_id, GRADE FROM `yhcr-prd-phm-bia-core.CB_FDM_DepartmentForEducation.src_KS4_exam` WHERE person_id IN {} order by person_id""".format(Pid)

sqlWideFormat1 = """ SELECT a.person_id,a.LSOA, a.AgeAtStartOfAcademicYear,a.FSMEligible,a.EverFSM6,a.EverFSM6P,a.EverFSMAll,
                                        a.EYPPE,a.EYPPBF,a.EYUEntitlement,
                                        a.EYEEntitlement,a.PPEntitlement,a.SBEntitlement, b.LDDFlag, b.SENDFlag , a.language,
                                        a.InCare, a.InCareAtCurrentSchool, a.NSiblings, a.birthorder FROM 
                                        `yhcr-prd-phm-bia-core.CB_FDM_DepartmentForEducation.src_census` a, 
                                        `yhcr-prd-phm-bia-core.CB_FDM_DepartmentForEducation.src_NCCIS` b
                                        WHERE a.person_id=b.person_id and b.person_id IN {} order by a.person_id, a.CensusDate""".format(Pid)


# In[106]:


tableDBWideFormat1 = pdg.read_gbq(sqlWideFormat1, dialect='standard')


# In[107]:


tableDBWideFormat1.person_id.unique()


# <h3> 8231 students records fetched from census table </h3>

# In[108]:


censusRecordsNEET = tableDBWideFormat1[["person_id","LSOA","AgeAtStartOfAcademicYear","FSMEligible","EverFSM6","EverFSM6P","EverFSMAll",
                                        "EYPPE","EYPPBF","EYUEntitlement",
                                        "EYEEntitlement","PPEntitlement","SBEntitlement","LDDFlag","SENDFlag","language", 
                                        "InCare","InCareAtCurrentSchool","NSiblings","birthorder" ]]
censusRecordsNEET


# In[109]:


from statistics import mode
censusRecordsNEET["FreeSchoolMeal"] = censusRecordsNEET.groupby('person_id').EverFSM6.transform('max') | censusRecordsNEET.EverFSM6P | censusRecordsNEET.EverFSMAll

censusRecordsNEET["EntitlementFlag"] = censusRecordsNEET.groupby('person_id').EYUEntitlement.transform('any') | censusRecordsNEET.EYEEntitlement | censusRecordsNEET.PPEntitlement | censusRecordsNEET.SBEntitlement
censusRecordsNEET["LDDFlag"] = censusRecordsNEET['LDDFlag'].apply(lambda set_: False if pd.isna(set_)== True else True)
censusRecordsNEET["SENDFlag"] = censusRecordsNEET['SENDFlag'].apply(lambda set_: False if ((pd.isna(set_)== True) | ((set_)=='N')) else True)

censusRecordsNEET["language"] = censusRecordsNEET.groupby('person_id').language.transform(mode) 
censusRecordsNEET["LDDFlag"] = censusRecordsNEET['LDDFlag'].apply(lambda set_: False if pd.isna(set_)== True else True)

# censusRecordsNEET["InCare"] = censusRecordsNEET['InCare'].apply(lambda set_: False if pd.isna(set_)== True else True)

# censusRecordsNEET["InCareAtCurrentSchool"] = censusRecordsNEET['InCareAtCurrentSchool'].apply(lambda set_: False if pd.isna(set_)== True else True)

# censusRecordsNEET["NSiblings"] = censusRecordsNEET['NSiblings'].apply(lambda set_: 0 if pd.isna(set_)== 0 else set_)
# censusRecordsNEET["birthorder"] = censusRecordsNEET['birthorder'].apply(lambda set_: 0 if pd.isna(set_)== 0 else set_)



censusRecordNEETDF = censusRecordsNEET[["person_id","LSOA","AgeAtStartOfAcademicYear","FSMEligible","FreeSchoolMeal","EntitlementFlag", "LDDFlag","SENDFlag","language",
                                        "InCare","InCareAtCurrentSchool","NSiblings","birthorder"]]

censusRecordNEETDF


# In[110]:


from statistics import mode
censusRecordsNEETLSOA15GP = censusRecordsNEET.query("AgeAtStartOfAcademicYear == 15").groupby(["person_id"]).agg({
        'LSOA':lambda x: x.dropna().tail(1),
        'FSMEligible': 'max', 
        'FreeSchoolMeal': 'max', 
    #    'EntitlementFlag': lambda x: x.dropna().tail(1)
        'EntitlementFlag': 'max',
        'LDDFlag':'max',
        'SENDFlag':'max',
        'language':mode,
        'InCare':mode,
        'InCareAtCurrentSchool':mode,
        'NSiblings':mode,
        'birthorder':mode
    })

censusRecordsNEETLSOA15GP=censusRecordsNEETLSOA15GP.reset_index()
censusRecordsNEETLSOA15GP


# In[111]:


#peopleWithLSOA15 = censusRecordsNEETLSOA15GP[(censusRecordsNEETLSOA15GP["LSOA"]!='[]')]
#peopleWithLSOA15
#peopleWithLSOA15 = censusRecordsNEETLSOA15GP

LSOAUniq = censusRecordsNEETLSOA15GP['LSOA'].apply(lambda x: str(x))
censusRecordsNEETLSOA15GP['LSOA']= censusRecordsNEETLSOA15GP['LSOA'].apply(lambda x: str(x))


# In[112]:


LSOA_subset_codes = tuple(LSOAUniq)
#LSOA_subset_codes


# In[113]:


QUERY_BY_LSOA_SUBSET =  """ SELECT a.LSOA_code as LSOA, a.LSOA_name, b.ward_name, a.geometry as geometry_home, 
a.lat_long FROM `yhcr-prd-phm-bia-core.CB_LOOKUPS.tbl_lsoa_boundaries` a,`yhcr-prd-phm-bia-core.CB_LOOKUPS.tbl_bradford_map_prep` b WHERE a.LSOA_code = b.LSOA_code and a.LSOA_code IN {}""".format(LSOA_subset_codes[1:-1])


# In[114]:


tableLSOA = pdg.read_gbq(QUERY_BY_LSOA_SUBSET, dialect='standard')


# In[115]:


GraphingGLDwithLSOA = pd.merge(censusRecordsNEETLSOA15GP,tableLSOA,on='LSOA',how='left')
GraphingGLDwithLSOA


# In[116]:


# modelDF.columns


# In[117]:


DFafterMerge = pd.merge(modelDF,GraphingGLDwithLSOA,on='person_id',how='left')
#DFafterMerge.drop(['index'],axis=1, inplace=True)
#DFafterMerge


# In[118]:


# DFafterMerge.columns


# In[119]:


DFafterMerge['PSEAS1Fail'] = DFafterMerge['PSEAS1Fail'].apply(lambda set_: False if pd.isna(set_)== True else set_)
DFafterMerge['PSEAS2Fail'] = DFafterMerge['PSEAS2Fail'].apply(lambda set_: False if pd.isna(set_)== True else set_)
DFafterMerge['PSEAS3Fail'] = DFafterMerge['PSEAS3Fail'].apply(lambda set_: False if pd.isna(set_)== True else set_)


DFafterMerge['CLLAS1Fail'] = DFafterMerge['CLLAS1Fail'].apply(lambda set_: False if pd.isna(set_)== True else set_)
DFafterMerge['CLLAS2Fail'] = DFafterMerge['CLLAS2Fail'].apply(lambda set_: False if pd.isna(set_)== True else set_)
DFafterMerge['CLLAS3Fail'] = DFafterMerge['CLLAS3Fail'].apply(lambda set_: False if pd.isna(set_)== True else set_)
DFafterMerge['CLLAS4Fail'] = DFafterMerge['CLLAS4Fail'].apply(lambda set_: False if pd.isna(set_)== True else set_)

DFafterMerge['PSRNAS1Fail'] = DFafterMerge['PSRNAS1Fail'].apply(lambda set_: False if pd.isna(set_)== True else set_)
DFafterMerge['PSRNAS2Fail'] = DFafterMerge['PSRNAS2Fail'].apply(lambda set_: False if pd.isna(set_)== True else set_)
DFafterMerge['PSRNAS3Fail'] = DFafterMerge['PSRNAS3Fail'].apply(lambda set_: False if pd.isna(set_)== True else set_)

DFafterMerge['RKUWFail'] = DFafterMerge['RKUWFail'].apply(lambda set_: False if pd.isna(set_)== True else set_)
DFafterMerge['RIPDFail'] = DFafterMerge['RIPDFail'].apply(lambda set_: False if pd.isna(set_)== True else set_)
DFafterMerge['RICDFail'] = DFafterMerge['RICDFail'].apply(lambda set_: False if pd.isna(set_)== True else set_)

DFafterMerge['FreeSchoolMeal'] = DFafterMerge['FreeSchoolMeal'].apply(lambda set_: False if pd.isna(set_)== True else set_)
DFafterMerge['LDDFlag'] = DFafterMerge['LDDFlag'].apply(lambda set_: False if pd.isna(set_)== True else set_)
DFafterMerge['SENDFlag'] = DFafterMerge['SENDFlag'].apply(lambda set_: False if pd.isna(set_)== True else set_)
DFafterMerge['InCare'] = DFafterMerge['InCare'].apply(lambda set_: False if pd.isna(set_)== True else set_)
DFafterMerge['InCareAtCurrentSchool'] = DFafterMerge['InCareAtCurrentSchool'].apply(lambda set_: False if pd.isna(set_)== True else set_)
DFafterMerge = DFafterMerge.rename(columns={'LSOA_y':'LSOA','LSOA_name_y':'LSOA_name','lat_long_y':'lat_long'})


# In[120]:


from statistics import mode
Flag1DataLSOA = DFafterMerge.groupby(['AcademicYear','newGLD','Gender']).agg({
    'Gender':'value_counts',
    'ever_NEET': lambda x: (x==False).sum(),
    #'Persistent_NEET':'sum',
    'Bradford_YN':'sum',
    'FreeSchoolMeal':'sum',
    #'LDDFlag':'sum',
    'SENDFlag': lambda x: (x==True).sum(),
    'language': mode,
    'InCare': lambda x: (x==True).sum(),
    'InCareAtCurrentSchool': lambda x: (x==True).sum(),
    'NSiblings':mode,
    'birthorder':mode
    #'newGLD':'value_counts',
     }).rename(columns={'Gender':'COUNTByGender', 'ever_NEET':'Non NEET'})
Flag1DataLSOA 


# In[121]:


from statistics import mode
FlagDataLSOA = DFafterMerge.groupby(['AcademicYear','newGLD','Gender']).agg({
    'Gender':'value_counts',
    'ever_NEET': lambda x: (x==True).sum(),
    #'Persistent_NEET':'sum',
    'Bradford_YN':'sum',
    'FreeSchoolMeal':'sum',
    #'LDDFlag':'sum',
    'SENDFlag': lambda x: (x==True).sum(),
    'language': mode,
    'InCare': lambda x: (x==True).sum()
    #'newGLD':'value_counts',
     }).rename(columns={'Gender':'COUNTByGender', 'ever_NEET':'Ever NEET'})
FlagDataLSOA 


# In[122]:


FlagDataLSOA = FlagDataLSOA.reset_index()
GLDTrueGenderFlagLSOA = FlagDataLSOA[(FlagDataLSOA['newGLD']==True)]
GLDFalseGenderFlagLSOA = FlagDataLSOA[(FlagDataLSOA['newGLD']==False)]
GLDTrueGenderFlagLSOA = GLDTrueGenderFlagLSOA.set_index(['AcademicYear','Gender'])
#print(GLDTrueGenderFlagLSOA)
GLDFalseGenderFlagLSOA = GLDFalseGenderFlagLSOA.set_index(['AcademicYear','Gender'])
#print(GLDFalseGenderFlagLSOA)

from matplotlib import ticker
import matplotlib.pyplot as plt

GLDFalseGenderFlagLSOA.drop(GLDFalseGenderFlagLSOA.tail(2).index,
        inplace = True)

fx=GLDFalseGenderFlagLSOA.plot(kind='bar')
plt.ylabel('Count of people from Bradford LSOA')

plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
plt.gca().xaxis.set_tick_params(rotation=0)

for bar in fx.patches:
    height = bar.get_height()
    fx.text(bar.get_x() + bar.get_width() / 2.0, height, f' {height:.2f}', fontsize=10,
            ha='center')
    fx.set_title("count of GLD non attainment based out of Bradford LSOA")
plt.tight_layout()
plt.show()


# In[123]:


from statistics import mode
GLD_LSOA_format = DFafterMerge.groupby(['LSOA'],as_index=False).agg({
        'LSOA_name':'max',
        'ward_name':'max',
        'newGLD': lambda x: (x==False).sum(),
        'geometry_home':'max',
        'lat_long':'max',
        'Bradford_YN':'sum',
        'EntitlementFlag':'sum',
        'FreeSchoolMeal':'sum',
        'SENDFlag': lambda x: (x==True).sum(),
        'ever_NEET':'sum',
        'Persistent_NEET':'sum',
        'language':mode,
        'InCare': lambda x: (x==True).sum(),
        'person_id':'count'}) 
GLD_LSOA_format 


# In[124]:


lenNotBRADFORD = GLD_LSOA_format.query('Bradford_YN==0')
lenNotBRADFORD


# In[125]:


lenBRADFORD = GLD_LSOA_format.query('Bradford_YN>=1')
lenBRADFORD


# In[126]:


#GLD_LSOA_format = GLD_LSOA_format.dropna()
totalBradfordPeople = GLD_LSOA_format.Bradford_YN.sum()
print("totalBradfordPeople",totalBradfordPeople)
totalGLDFailBradford = GLD_LSOA_format.newGLD.sum()
print("totalGLDFailBradford",totalGLDFailBradford)
totalPersonIDBradford = GLD_LSOA_format.person_id.sum()
print("totalPersonIDBradford",totalPersonIDBradford)


# ### Count of People 7940 of which 6038 where from Bradford (76%) of which 5105 (64.29%) have failed GLD

# In[127]:


from statistics import mode
LLD = DFafterMerge.groupby(['LSOA','Gender'], as_index=False).agg({
    'person_id':'count',
    'newGLD': lambda x: (x==False).sum(),
    'ever_NEET':'sum',
    'Persistent_NEET':'sum',
    'language':mode,
    'InCare': lambda x: (x==True).sum(),
     'ward_name':'max'})
    
LLD = LLD.dropna()
LLGDraph = LLD.reset_index().sort_values(['newGLD','LSOA'], ascending=False ).query('newGLD>=1').head(18)


# In[128]:


LLGDraphIndex = LLGDraph.set_index(['ward_name','Gender'])
LLGDraphIndex = LLGDraphIndex.drop(['index','LSOA'],axis=1)
LLGDraphIndex


# In[129]:


#LLGDraphIndex = LLGDraphIndex.drop(['LSOA'],axis=1)


# In[130]:


LLGDraphIndexFemale = LLGDraphIndex.query("Gender.str.contains('F')")
LLGDraphIndexMale = LLGDraphIndex.query("Gender.str.contains('M')")
# print(LLGDraphIndexFemale)
# print(LLGDraphIndexMale)

from matplotlib import ticker
import matplotlib.pyplot as plt

plt.style.use('ggplot')
fig, (ax1,ax2) = plt.subplots(nrows=1,ncols=2,figsize=(24,12))
plt.subplots_adjust(wspace=0.2)

LLGDraphIndexFemale.plot(kind='bar',ax=ax1)
ax1.set_ylabel('count of GLD non attainment based out of Bradford LSOA ')

for bar in ax1.patches:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width() / 2.0, height, f' {height:.2f}', fontsize=10,
            ha='center')
    ax1.set_title("Top 10 Wards: GLD Failure Counts Gender wise")
    

LLGDraphIndexMale.plot(kind='bar', ax=ax2)
ax2.set_ylabel('count of GLD non attainment based out of Bradford LSOA ')
for bar in ax2.patches:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width() / 2.0, height, f' {height:.2f}', fontsize=10,
            ha='center')
    ax2.set_title("Top 10 Wards: GLD Failure Counts Gender wise")
    
plt.show() 


# ### 117/141 82% female and 224/279 80% male

# In[131]:


GLD_LSOA_format['CountofPeople'] = GLD_LSOA_format['person_id']
GLD_LSOA_format['CountofPeopleFailed'] = GLD_LSOA_format['newGLD']
GLD_LSOA_format = GLD_LSOA_format.dropna()
GLD_LSOA_format.sort_values(['newGLD','LSOA'], ascending=False )


# In[132]:


#! pip install geopandas
#! pip install cartopy
#! pip install contextily
#! pip install folium
#! pip install geojson


# In[133]:


import geopandas as gpd
import cartopy as ccrs
import contextily as cx
import folium
import geojson
from pprint import pprint


# In[134]:


bradford = folium.Map(tiles="CartoDB positron", location=(53.8313, -1.8431), zoom_start=10)


# In[135]:


GLD_LSOA_format['geometry'] = gpd.GeoSeries.from_wkt(GLD_LSOA_format['geometry_home'], crs=4258)
gdf_GLD_by_lsoa = gpd.GeoDataFrame(GLD_LSOA_format, geometry='geometry')


# In[136]:


bradfordChoropleth = folium.Choropleth(
   geo_data = gdf_GLD_by_lsoa,
   data = gdf_GLD_by_lsoa,
   columns= ['LSOA','CountofPeople','CountofPeopleFailed','ever_NEET','SENDFlag','FreeSchoolMeal','language','InCare'],
   key_on = "feature.properties.LSOA",
   fill_color="YlOrRd",
   fill_opacity="0.8",
   line_opacity="0.5",
   bins=[0,10,20,30,60,80,100,200],
   legend_name="Count of People GLD non Attainment, SEND status and Free Meal in Bradford Area",
    tooltip='CountofPeople'
).add_to(bradford)


# In[137]:


bradfordChoropleth.geojson.add_child(folium.features.GeoJsonTooltip(['ward_name', 'CountofPeople','CountofPeopleFailed','ever_NEET','SENDFlag','FreeSchoolMeal','language','InCare'], aliases=['Post Code: ','#of People: ', 'GLD Failed','Ever NEET','SEN','Free School Meal', 'First Language Code', 'In Care']))    
folium.LayerControl().add_to(bradford)
bradford


# In[287]:


print(DFafterMerge.ever_NEET.sum())
print(len(DFafterMerge.ever_NEET))


# model_covariates_flag = sm.formula.glm("ever_NEET ~ newGLD+SENDFlag)",
#                        family=sm.families.Binomial(), data=DFafterMerge).fit()

model_covariates_flag = sm.formula.glm("ever_NEET ~ newGLD",
                       family=sm.families.Binomial(), data=DFafterMerge).fit()

print(model_covariates_flag.summary())
predictionsCovaritesFlag=model_covariates_flag.predict()

# print(np.unique(predictionsCovaritesFlag))

unique, counts = np.unique(predictionsCovaritesFlag, return_counts=True)

# print(np.asarray((unique, counts)).T)

uniqueT, countsT = np.unique(modelDF['ever_NEET'], return_counts=True)

# print(np.asarray((uniqueT, countsT)).T)

predictions_CovaritesFlag = [1 if x < 0.3 else 0 for x in predictionsCovaritesFlag]
# print(model_covariates_flag.params.Intercept)
odds_ratio_covariatesFlag=np.exp(model_covariates_flag.params.Intercept)
# print('odds ratio is ',odds_ratio_covariatesFlag)

coefs = pd.DataFrame({
    'coef': model_covariates_flag.params.values,
    'odds ratio': np.exp(model_covariates_flag.params.values),
     'p-values': model_covariates_flag.pvalues
})
print(coefs)


# In[140]:


# cnf_matrix_CovaritesFlag = confusion_matrix(DFafterMerge["ever_NEET"], 
#                        predictions_CovaritesFlag)
# print(cnf_matrix_CovaritesFlag)

# print(classification_report(DFafterMerge["ever_NEET"], 
#                             predictions_CovaritesFlag, 
#                             digits = 3))

# class_names=["0","1"] # name  of classes
# fig, ax = plt.subplots()
# tick_marks = np.arange(len(class_names))
# plt.xticks(tick_marks, class_names)
# plt.yticks(tick_marks, class_names)
# # create heatmap
# sns.heatmap(pd.DataFrame(cnf_matrix_CovaritesFlag), annot=True, cmap="YlGnBu" ,fmt='g')
# ax.xaxis.set_label_position("top")
# plt.tight_layout()
# plt.title('Confusion matrix', y=1.1)
# plt.ylabel('Actual label NEET')
# plt.xlabel('Predicted label NEET')


# <div class="alert alert-block alert-info">
# 
# <h3> To understand the predictive power of becoming NEET from other covariates using Structural Equation Modelling 
# Structural equation modeling (SEM) is a multivariate statistical analysis technique that is used to analyze structural relationships between variables under study. This technique uses a combination of factor analysis as well as multiple regression, to analyze the structural relationship between measured variables and latent constructs. This method is preferred by researchers because it estimates the multiple and interrelated dependence in a single analysis.    
#     
#     
# </h3>
#     
# </div>

# In[141]:


# !pip install fsspec
# !pip install s3fs
# !pip install boto
# !pip install semopy


# In[142]:


import pandas as pd
import boto
import semopy
import warnings
from pandas.errors import SettingWithCopyWarning
warnings.simplefilter(action='ignore', category=(SettingWithCopyWarning))


# In[143]:


# DFafterMerge.columns
DFafterMerge = DFafterMerge.copy(deep=True)


# In[144]:


DFForSemopy = DFafterMerge[['person_id', 'Gender','AcademicYear',  'PSEAS1', 'PSEAS2', 'PSEAS3',
       'PSETotal', 'CLLAS1', 'CLLAS2', 'CLLAS3', 'CLLAS4', 'CLLTotal',
       'PSRNAS1', 'PSRNAS2', 'PSRNAS3', 'PSRNTotal', 'RKUW', 'RIPD', 'RICD',
       'EYFSPTotal','ever_NEET', 'Persistent_NEET', 'Total_neet_months',
       'total_number_of_observations', 'percentage_time_neet',
       'NumberOfMonthsUnknown', 'newGLD', 'AcademicBegin', 'AcademicEnd',
       'FreeSchoolMeal',
       'LDDFlag', 'SENDFlag', 'language', 'InCare']]


# In[145]:


DFForSemopy['language']= DFForSemopy.language.fillna("missing")


# In[201]:


DFForSemopy["LDDFlag"] = DFForSemopy['LDDFlag'].apply(lambda set_: 1 if (set_)== True else 0)
DFForSemopy["Persistent_NEET"] = DFForSemopy['Persistent_NEET'].apply(lambda set_: 1 if (set_)== True else 0)
DFForSemopy["newGLD"] = DFForSemopy['newGLD'].apply(lambda set_: 1 if (set_)== True else 0)
DFForSemopy["FreeSchoolMeal"] = DFForSemopy['FreeSchoolMeal'].apply(lambda set_: 1 if (set_)== True else 0)
DFForSemopy["SENDFlag"] = DFForSemopy['SENDFlag'].apply(lambda set_: 1 if (set_)== True else 0)
DFForSemopy["InCare"] = DFForSemopy['InCare'].apply(lambda set_: 1 if (set_)== True else 0)

from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(sparse_output=False)
Gender_encoded_data = encoder.fit_transform(DFForSemopy[['Gender']].to_numpy())
Gender_encoded_df = pd.DataFrame(Gender_encoded_data, columns=encoder.get_feature_names_out())

language_encoded_data = encoder.fit_transform(DFForSemopy[['language']].to_numpy())
language_encoded_df = pd.DataFrame(language_encoded_data, columns=encoder.get_feature_names_out())

DFForSemopyNew = pd.concat([DFForSemopy, Gender_encoded_df, language_encoded_df], axis=1)
DFForSemopyNew = DFForSemopyNew.drop(['Gender','language','AcademicYear'], axis=1)
# pd.set_option('display.max_columns', None)
# DFForSemopyNew


# <div class="alert alert-block alert-info">
# <h3>This code iterates through each domain in GLD, creating a separate linear regression model using statsmodels.api.OLS.
# The summary output provides model statistics and individual feature coefficients for interpretation.
# </h3>
# /div>

# In[271]:


import statsmodels.api as sm

# Define target variable
outcome_var = "ever_NEET"


domains = 'newGLD','PSEAS1','PSEAS2','PSEAS3','CLLAS1','CLLAS2','CLLAS3','CLLAS4','PSRNAS1','PSRNAS2','PSRNAS3','RKUW','RIPD','RICD','EYFSPTotal','PSETotal','CLLTotal' 
all_r_squared = []
all_adjusted_r_squared = []
all_rmse = []
all_num_features = []

all_coefficients = []
all_std_errors = []
all_p_values = []

# Loop through each domain
for domain in domains:
    domain_features = [feature for feature in DFForSemopyNew.columns if feature.startswith(domain)]

    print(domain_features)
    # Build and fit the model
    model = sm.OLS(DFForSemopyNew[outcome_var], DFForSemopyNew[domain_features])
    results = model.fit()

    # Extract significant features (consider your significance threshold)
    pvals = results.pvalues  # Get p-values from the results object
    significant_features = pvals[pvals < 0.05].index  # Identify features with p-value < 0.05

    # Print summary results
    print(f"\n*** {domain.upper()} Domain Regression Results ***")
    print(results.summary())
    
    # R-squared
    r_squared = results.rsquared
    all_r_squared.append(r_squared)
    
    # Adjusted R-squared
    adjusted_r_squared = results.rsquared_adj
    all_adjusted_r_squared.append(adjusted_r_squared)

    # Root Mean Squared Error (RMSE)
    # 1. Calculate predicted values
    predicted_y = results.predict(model.exog)

    # 2. Calculate squared residuals
    squared_residuals = (predicted_y - model.endog)**2

    # 3. Calculate average squared residual (mean)
    mean_squared_residual = squared_residuals.mean()

    # 4. Take the square root to get RMSE
    rmse = np.sqrt(mean_squared_residual)
    all_rmse.append(rmse)

    # Number of features (excluding intercept)
    num_features = len(results.params) - 1  # Remove intercept
    all_num_features.append(num_features)
    
    coefficients = results.params
    all_coefficients.append(coefficients)
    std_errors = results.bse
    all_std_errors.append(std_errors)
    p_values = results.pvalues
    all_p_values.append(p_values)


# In[283]:


from tabulate import tabulate

# Sample data (replace with your actual results)
data = {
    "Domain": domains,
    "R-squared": all_r_squared,   
    "RMSE": all_rmse,   
    "Coefficients":all_coefficients,
    "Standard Errors ": all_std_errors,
    "P-values":all_p_values
}

#"Adjusted R-squared": all_adjusted_r_squared,
#"Num Features": all_num_features
df = pd.DataFrame(data)

# # Sort by R-squared (descending)
#df_sorted = df.sort_values(by="Coefficients", ascending=False)

# # Print the table with formatting
table = tabulate(df, headers="keys", tablefmt="fancy_grid")

print(type(table))
# import openpyxl
# table.to_excel("ols_model_output.xlsx",
#               sheet_name='Statistical_significance', index=False) 


# from openpyxl import Workbook

# # Create a workbook and worksheet
# wb = Workbook()
# ws = wb.active

# # Write the table data to the worksheet
# ws.append(tabulate(df, headers="keys", tablefmt="fancy_grid").splitlines())

# # Save the workbook
# wb.save("ols_model_output.xlsx")

print(table)


# In[ ]:





# In[284]:


import statsmodels.api as sm

# Define target variable
outcome_var = "ever_NEET"


domains = 'newGLD','PSEAS1','PSEAS2','PSEAS3','CLLAS1','CLLAS2','CLLAS3','CLLAS4','PSRNAS1','PSRNAS2','PSRNAS3','RKUW','RIPD','RICD','EYFSPTotal','PSETotal','CLLTotal' 


all_coefficients = []
all_odds_ratio = []
all_p_values = []

# Loop through each domain
for domain in domains:
    domain_features = [feature for feature in DFForSemopyNew.columns if feature.startswith(domain)]
    
    
    formula = f"ever_NEET ~ {' + '.join(domain_features)}"  
    model_covariates_flag = sm.formula.glm(formula,
                       family=sm.families.Binomial(), data=DFForSemopyNew).fit()

    print(model_covariates_flag.summary())
    predictionsCovaritesFlag=model_covariates_flag.predict()

    # print(np.unique(predictionsCovaritesFlag))

    unique, counts = np.unique(predictionsCovaritesFlag, return_counts=True)

    # print(np.asarray((unique, counts)).T)

    uniqueT, countsT = np.unique(modelDF['ever_NEET'], return_counts=True)

    # print(np.asarray((uniqueT, countsT)).T)

    predictions_CovaritesFlag = [1 if x < 0.3 else 0 for x in predictionsCovaritesFlag]
    # print(model_covariates_flag.params.Intercept)
    odds_ratio_covariatesFlag=np.exp(model_covariates_flag.params.Intercept)
    # print('odds ratio is ',odds_ratio_covariatesFlag)
    
    all_coefficients.append(model_covariates_flag.params.values)
    all_odds_ratio.append(np.exp(model_covariates_flag.params.values))
    all_p_values.append(model_covariates_flag.pvalues)
    


# In[286]:


from tabulate import tabulate

# Sample data (replace with your actual results)
OLSGLMdata = {
    "Domain": domains,
    "Coefficients":all_coefficients,
    "Odds Ratio ": all_odds_ratio,
    "P-values":all_p_values
}

#"Adjusted R-squared": all_adjusted_r_squared,
#"Num Features": all_num_features
df = pd.DataFrame(OLSGLMdata)

# # Sort by R-squared (descending)
#df_sorted = df.sort_values(by="Coefficients", ascending=False)

# # Print the table with formatting
table = tabulate(df, headers="keys", tablefmt="fancy_grid")

print(table)


# <div class="alert alert-block alert-info">
# <h2>Interpreting SEM model results from semopy: Estimate, Std. Err, z-value, and p-value
# When analyzing a SEM model in semopy, you'll typically encounter four key output values for each parameter:
# </h2>
# 
# <li><emphasis>Estimate:</emphasis> This reflects the magnitude and direction of the relationship between two variables in the model. Positive values indicate a positive relationship, while negative values indicate a negative relationship.</li>
# <li><emphasis>Std. Err:</emphasis> This stands for Standard Error, which represents the estimated variability of the Estimate. A smaller Std. Err. indicates higher precision in the estimate.</li>
# <li><emphasis>z-value:</emphasis> This is a standardized version of the Estimate, calculated by dividing it by its Std. Err. It reflects the number of standard deviations the Estimate is away from zero.</li>
# <li><emphasis>p-value:</emphasis> This represents the probability of observing an Estimate as extreme as the one obtained, assuming no true relationship exists between the variables. Smaller p-values indicate stronger evidence against the null hypothesis (no relationship).</li>
# 
# Here's how to interpret these values:
# 
# <li><emphasis> Interpreting Estimates:</emphasis>
# 
# Look for large and consistent Estimates in line with your theoretical expectations.
# Small and non-significant Estimates (see below) suggest the relationship is weak or not present.</li>
#     
# <li><emphasis>Interpreting Std. Err.:</emphasis>
# 
# Smaller Std. Err. indicates more precise estimates and higher confidence in the Estimate.
# Larger Std. Err. indicates more uncertainty and less confidence in the Estimate.</li>
# 
# <li><emphasis>Interpreting z-value:</emphasis>
# 
# Absolute values greater than 1.96 (95% confidence level) or 2.58 (99% confidence level) are generally considered statistically significant.
# Higher positive z-values indicate stronger positive relationships.
# Higher negative z-values indicate stronger negative relationships.</li>
#     
# <li><emphasis>Interpreting p-value:</emphasis>
# 
# Smaller p-values (typically less than 0.05) indicate statistically significant relationships. This means it's unlikely the observed relationship is due to chance.
# Larger p-values indicate non-significant relationships, suggesting the observed relationship could be due to chance.</li>
# 
# <li><emphasis>Remember:</emphasis>
# 
# Consider all four values together for a comprehensive interpretation.
# Statistical significance alone doesn't guarantee practical importance. Evaluate the magnitude of the Estimate as well.
# Consider the overall model fit and other model diagnostics to ensure your results are reliable.</li>
# </div>

# In[202]:


# all_zero_columns = [col for col in DFForSemopyNew.columns if DFForSemopyNew[col].sum() == 0.0]

# if all_zero_columns:
#     print("Columns with all zeros:", all_zero_columns)
# else:
#     print("No columns with all zeros found.")


# In[248]:


# pattern = "x0_"

# new_col_names = [
#     f"{col} " if col.startswith(pattern) else col
#     for col in DFForSemopyNew.columns
# ]
# new_col_names


# In[220]:


model_spec = """
  # measurement model
    GLD =~  newGLD + PSEAS1 + PSEAS2 +  PSEAS3+ CLLAS1 + CLLAS2 + CLLAS3 + CLLAS4 + PSRNAS1 + PSRNAS2 + PSRNAS3 + RKUW + RIPD + RICD 
    Social =~ SENDFlag + FreeSchoolMeal +LDDFlag +x0_F + x0_M
    Language =~ x0_AFK+x0_AKAT+x0_ALB+x0_AMR+x0_ARA+x0_ARAI+x0_BNG+x0_BNGA+x0_BNGS+x0_BSL+x0_CCF+x0_CHI+x0_CHIC+x0_CHIM+x0_CWA+x0_CZE+x0_DUT+x0_ENB+x0_ENG+x0_FRN+x0_GEO+x0_GRE+x0_GUJ+x0_HDK+x0_HGR+x0_HIN+x0_ISL+x0_KIN+x0_KNK+x0_KUR+x0_LIN+x0_LTV+x0_MAR+x0_NDB+x0_NDBS+x0_NDBZ+x0_NEP+x0_NOT+x0_OTB+x0_OTH+x0_OTL+x0_PAT+x0_PHR+x0_PNJ+x0_PNJA+x0_PNJG+x0_PNJM+x0_PNJP+x0_POL+x0_POR+x0_PORA+x0_PRS+x0_PRSA+x0_RUS+x0_SCB+x0_SCBC+x0_SHO+x0_SLO+x0_SOM+x0_SPA+x0_STS +x0_SWA+x0_SWAA+x0_TAM + x0_TEL+x0_TGL+x0_TGLF +x0_TGLG+x0_THA+x0_TUR+x0_UKR+x0_URD+x0_VIE+x0_YOR+x0_ZZZ+x0_missing
  # regressions
    ever_NEET ~ GLD + Social +Language
"""

# Instantiate the model
model = semopy.Model(model_spec)

# Fit the model using the data
model.fit(DFForSemopyNew)

dfModelOutput = model.inspect()

import openpyxl
dfModelOutput.to_excel("Semopy_model_output.xlsx",
              sheet_name='Statistical significance', index=False) 




# # Show the results using the inspect method
# with pd.option_context('display.max_rows', None, 'display.max_columns', None):


# In[194]:


pd.set_option('display.max_columns', None)


# In[251]:


warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# Define target variable
target_variable = "ever_NEET"


column_to_include = DFForSemopyNew.drop(['ever_NEET','Total_neet_months','Persistent_NEET','total_number_of_observations',
 'percentage_time_neet','NumberOfMonthsUnknown','person_id'],axis=1).columns


# List all columns except the one to exclude
#columns_except_one = DFForSemopyNew.drop(column_to_exclude, axis=1).columns.tolist()



X_train, X_test, y_train, y_test = train_test_split(
    DFForSemopyNew[column_to_include], DFForSemopyNew[target_variable], test_size=0.2, random_state=42
)

# Define XGBoost model parameters
model_params = {
    "objective": "reg:logistic",  # Use "reg:logistic" for regression-based prediction
    "max_depth": 3,
    "learning_rate": 0.1,
    "n_estimators": 100,
}

# Initialize and train the model
model = XGBClassifier(**model_params)
model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = model.predict_proba(X_test)[:, 1]  # Get probability of becoming NEET

# Evaluate model performance (e.g., accuracy, AUC-ROC)
# Consider using libraries like `sklearn.metrics` for evaluation

# Analyze feature importance
feature_importance = model.feature_importances_
# Visualize or interpret feature importance

# Draw conclusions and discuss limitations

# Optional: Save the model for future use
model.save_model("model.xgb")


# In[252]:


from xgboost import plot_importance

plot_importance(model, max_num_features=20)  # Show top 10 features


# In[ ]:




