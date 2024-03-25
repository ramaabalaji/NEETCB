#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Check free memory available
get_ipython().run_line_magic('system', 'free -m')


# In[ ]:


#!pip install jupyterlab


# In[ ]:


#! pip install pandas_gbq

#! pip install import-ipynb #to import from other notebook files


# <!-- ![Screenshot 2023-12-09 at 15.33.55.png](attachment:8964a64f-ffe2-4f66-bf88-58f65ae4c048.png) -->

# In[ ]:


# Import required libraries
from google.cloud import bigquery
import gc
from dateutil.relativedelta import relativedelta
import numpy as np
import math
import os
import pandas_gbq as pdg
import pandas as pd
import matplotlib.pyplot as plt
#import import_ipynb
#import NCCISCohortEDA as nccis
import kaleido
import plotly.graph_objects as go
import plotly
# Interactive Plotly map 
from matplotlib import style
import seaborn as sns
import plotly.graph_objects as go
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
from shapely.geometry import Point, Polygon
import plotly.express as px
import warnings
style.use('fivethirtyeight')
warnings.filterwarnings("ignore")


# In[ ]:


#PLOT_OUTPUT_LOCATION = "reports/figures/"
def returnPlotLocation():
    PLOT_OUTPUT_LOCATION = "reports/figures/"
    return PLOT_OUTPUT_LOCATION


# In[ ]:


def retINTERIM_DATA_LOCATION():
    INTERIM_DATA_LOCATION = "data/interim/"
    return INTERIM_DATA_LOCATION


# In[ ]:


def everNeetObservation():
    return pd.read_csv(retINTERIM_DATA_LOCATION() + "EverNEET16_18YAge.csv")


# In[ ]:


def PersistantNeetMoreThan12():
    return pd.read_csv(retINTERIM_DATA_LOCATION() + "PersistantNEET16_18YAge.csv")


# <div class="alert alert-block alert-warning">
#     <H2># Records are retrieved and filtered and taken from NEETCohort16To18.ipynb # </H2> <br>
#     - Use appropriate functions to retrive dataframes of your interest and perform the necessary task call everNEETObservation()
# </div>

# In[ ]:


everNEETCohort = everNeetObservation()


# <div class="alert alert-block alert-warning">
#      <H3> #This file works on NEET COHORT Explicitly #</H3> 
#     
# -- You can retrive various dataframe as needed for researches  
#     1. <b> notInCensusPeople() </b> returns people who do not have records in Census table. If you want to make any research on them you can use this function to retrun the values of person_id <br>
# 2. <b>PersistantNeetMoreThan12() </b> - People who have been persistently NEET for more than 12 months Duration use this Dataframe <br>
# 3. <b>everNEETObservation()</b> - People have atleast 1 NEET Observation until 8 Observations use this Dataframe<br>
#     4. <b>peopleNotInCensusLSOA15()</b> - People who do not have LSOA listed when their age was 15
# </div>
# 
# 

# <div class="alert alert-block alert-info">
# <h3> # Ever NEET threshold is Total NEET Observation is more than 1 upto less than 8 observations # </h3> <br>
#     - use the dataframe <b>everNEETObservation</b>
# </div>

# In[ ]:


everNEETCohort


# # Selecting NEET COHORT by  filtering records having even one NEET Observations for a person

# <div class="alert alert-block alert-info">
#     <h3> # Persistent NEET threshold is No of months being NEET over 12 # </h3> <br>
#     - use the dataframe <b> PersistantNeetMoreThan12</b>
# </div>

# In[ ]:


PersistantNeetMoreThan12 = PersistantNeetMoreThan12()


# In[ ]:


cnt = PersistantNeetMoreThan12["person_id"].unique()
len(cnt)


# <div class="alert alert-block alert-success">
# <h2> # For what Activity codes the NEET Observations are high and how many people are being NEET under this category #</h2>
# </div>

# In[ ]:


observationNeetMoreThan1gp = PersistantNeetMoreThan12.groupby(["CurrentActivityCode"]).agg(
    {'total_neet_observations': 'max', 'No_of_months_neet': 'max', 'person_id':'nunique'})
observationNeetMoreThan1gp=observationNeetMoreThan1gp.reset_index()
observationNeetMoreThan1gp


# In[ ]:


fig = plt.figure()
# Remove vertical space between axes
#fig.subplots_adjust(hspace=0)

plt.rcParams['text.color'] = 'Black'
plt.rcParams['font.size'] = 8

ax = observationNeetMoreThan1gp.plot(kind='bar', x='CurrentActivityCode', y='person_id', label='Person Activity Code', color='pink')
ax.bar_label(ax.containers[0])
ax.set_xlabel('Activity Codes')
ax.set_ylabel('Count of people')
ax.legend(loc='upper left', fontsize= 10)

plt.savefig(returnPlotLocation() + "PersistantNEET16to18/ActivityCodesVsCount", dpi=300)

bx = observationNeetMoreThan1gp.plot(kind='bar', x='CurrentActivityCode', y='No_of_months_neet', label='Neet months in that Activity Code', color='cyan')
bx.bar_label(bx.containers[0])
bx.set_xlabel('Activity Codes')
bx.set_ylabel('Duration spent as NEET in that Activity Code')
bx.legend(loc='upper left', fontsize= 10)

plt.savefig(returnPlotLocation() + "PersistantNEET16to18/ActivityCodesVsDuration", dpi=300)


# <div class="alert alert-block alert-success">
# <h2># Search in Census table for relevant data about these NEET people to understand their demograph #</h2><br>
# 1. Check for the LSOA if the last value is null take the available latest values for that person <br> 
# 2. Check if any FSMeligible, EverFSM6, EverFSM6P, EverFSMAll - Free School Meal <br>
# 3. InCareAtCurrentSchool Child in care <br>
# 4. Disability<br>
# 5. EYPPE, EYPPBF <br>
# 6. LSOA<br>
# 7. EYUEntitlement, EYEEntitlement, PPEntitlement, SBEntitlement Any Entitlements
# </div>
# 

# In[ ]:


personUniqueID = PersistantNeetMoreThan12["person_id"].unique()
ct=len(personUniqueID)
ct


# In[ ]:


#pID = map(str, personUniqueID) 
Pid = tuple(personUniqueID)


# In[ ]:


sqlWideFormat = """ SELECT *
FROM `yhcr-prd-phm-bia-core.CB_FDM_DepartmentForEducation.src_census` 
WHERE person_id IN {} order by person_id, CensusDate""".format(Pid)

#sqlWideFormat


# In[ ]:


tableDBPersistantNEETLSOA = pdg.read_gbq(sqlWideFormat, dialect='standard')



# In[ ]:


tableDBPersistantNEETLSOA


# In[ ]:


# column_names = list(tableDBWideFormat1.columns)
# column_names


# In[ ]:


censusRecordsPersistantNEET = tableDBPersistantNEETLSOA[["person_id","CensusDate","LSOA","AgeAtStartOfAcademicYear","FSMEligible","EverFSM6","EverFSM6P","EverFSMAll",
                                        "InCareAtCurrentSchool","Disability","EYPPE","EYPPBF","EYUEntitlement",
                                        "EYEEntitlement","PPEntitlement","SBEntitlement"]]


# In[ ]:


censusRecordsPersistantNEET


# In[ ]:


censusRecordsPersistantNEET.AgeAtStartOfAcademicYear.unique()


# In[ ]:


censusRecordsPersistantNEET.person_id.unique()


# In[ ]:


x = censusRecordsPersistantNEET.query("AgeAtStartOfAcademicYear >=0 & AgeAtStartOfAcademicYear <= 15")
x.person_id.unique()


# <div class="alert alert-block alert-success">
# 
# <h2># Findings :- # </h2> <br>
# 1. No of  Persistant NEET COHORT 378 <br>
# 2. No of people recorded in census table 373 - Loosing 5 records <br>
# 3. No of People with LSOA listed when they were 15 - 341 (will loose data for 37 people) <br>
# 
#     
# <b> # Finding Those Cohort who are present as NEET but do not have even one record in Census table notInCensusPeople #</b>
# </div>

# In[ ]:


PersistantPeopleInCensus=pd.DataFrame(censusRecordsPersistantNEET["person_id"].unique(), columns=['person_id']).sort_values('person_id')
PersistantPeopleInNEETCohort=pd.DataFrame(personUniqueID,columns=['person_id'],).sort_values('person_id')
PersistantPeopleInCensus.set_index('person_id', inplace=True)
PersistantPeopleInNEETCohort.set_index('person_id', inplace=True)

notInCensusPersistantPeople=PersistantPeopleInNEETCohort[~PersistantPeopleInNEETCohort.index.isin(PersistantPeopleInCensus.index)]
# #df2[df1 != df2].dropna(how='all', axis=1).dropna(how='all', axis=0)
# len(dftest2)
#dftest3.dropna(inplace=True)
notInCensusPersistantPeople


# In[ ]:


censusRecordsPersistantNEET["FreeSchoolMeal"] = censusRecordsPersistantNEET.groupby('person_id').EverFSM6.transform('any') | censusRecordsPersistantNEET.EverFSM6P | censusRecordsPersistantNEET.EverFSMAll

censusRecordsPersistantNEET["EntitlementFlag"] = censusRecordsPersistantNEET.groupby('person_id').EYUEntitlement.transform('any') | censusRecordsPersistantNEET.EYEEntitlement | censusRecordsPersistantNEET.PPEntitlement | censusRecordsPersistantNEET.SBEntitlement

censusRecordsPersistantNEET = censusRecordsPersistantNEET[["person_id","CensusDate","AgeAtStartOfAcademicYear","LSOA","FSMEligible","InCareAtCurrentSchool","Disability","FreeSchoolMeal","EntitlementFlag"]]

#censusRecordsPersistantNEET


# In[ ]:


#censusRecordNEETDF.query("person_id == 13764032")


# In[ ]:


PersistantCensusNEETLSOA15GP = censusRecordsPersistantNEET.query("AgeAtStartOfAcademicYear == 15").groupby(["person_id"]).agg({
        'LSOA':lambda x: x.dropna().tail(1),
        'FSMEligible': lambda x: x.dropna().tail(1), 
        'InCareAtCurrentSchool': lambda x: x.dropna().tail(1),
        'Disability': lambda x: x.dropna().tail(1),
        'FreeSchoolMeal': lambda x: x.dropna().tail(1), 
        'EntitlementFlag': lambda x: x.dropna().tail(1)
    })

PersistantCensusNEETLSOA15GP=PersistantCensusNEETLSOA15GP.reset_index()
PersistantCensusNEETLSOA15GP


# <div class="alert alert-block alert-success">
#     # If you do not wish to limit the LSOA at 15 use the below block of code#
# </div>

# In[ ]:


PersistantPeopleWithLSOA15 = PersistantCensusNEETLSOA15GP[(PersistantCensusNEETLSOA15GP["LSOA"]!='[]')]
PersistantPeopleWithLSOA15


# In[ ]:


PersistantPeopleInCensusNEET=pd.DataFrame(PersistantCensusNEETLSOA15GP["person_id"].unique(), columns=['person_id']).sort_values('person_id')
PersistantPeopleInCohortLSOA15=pd.DataFrame(PersistantPeopleWithLSOA15["person_id"].unique(), columns=['person_id']).sort_values('person_id')
PersistantPeopleInCensusNEET.set_index('person_id', inplace=True)
PersistantPeopleInCohortLSOA15.set_index('person_id', inplace=True)
PersistantPeopleNotInCensusLSOA15=PersistantPeopleInCensusNEET[~PersistantPeopleInCensusNEET.index.isin(PersistantPeopleInCohortLSOA15.index)]
PersistantPeopleNotInCensusLSOA15


# <div class="alert alert-block alert-success">
# <b># There are close to 31 records than has missing LSOA when they were 15. Retrieve these people using peopleNotInCensusLSOA</b>
# </div>
# 

# In[ ]:


PersistantPeopleWithLSOA15.info()

LSOAUniq = PersistantPeopleWithLSOA15['LSOA'].apply(lambda x: str(x))
PersistantPeopleWithLSOA15['LSOA']= PersistantPeopleWithLSOA15['LSOA'].apply(lambda x: str(x))


# # For all the LSOA listed get its corresponding details from the lookup tbl_lsoa_boundaries

# In[ ]:


LSOA_subset_codes = tuple(LSOAUniq)


# In[ ]:


QUERY_BY_LSOA_SUBSET =  """ SELECT LSOA_code as LSOA, LSOA_name, geometry as geometry_home, 
lat_long FROM `yhcr-prd-phm-bia-core.CB_LOOKUPS.tbl_lsoa_boundaries` WHERE LSOA_code IN {}""".format(LSOA_subset_codes[1:-1])


# In[ ]:


#QUERY_BY_LSOA_SUBSET


# In[ ]:


tablePersistantLSOA = pdg.read_gbq(QUERY_BY_LSOA_SUBSET, dialect='standard')
tablePersistantLSOA1 = tablePersistantLSOA


# In[ ]:


#tableLSOA


# In[ ]:


#censusRecordNEETwithLSOA = pd.merge(censusRecordsNEETGP,tableLSOA,on='LSOA',how='left')
PersistantCensusRecordNEETwithLSOA = pd.merge(PersistantPeopleWithLSOA15,tablePersistantLSOA,on='LSOA',how='left')
#censusRecordNEETwithLSOA


# In[ ]:


PersistantNeetMoreThan12BFMerge = PersistantNeetMoreThan12[["person_id","CurrentActivityCode","year_month_birth","Age","gender","ethinicity","academic_start_date","total_neet_observations","custody_offender_observations","status_unknown_observations"]].sort_values("person_id")


# In[ ]:


PersistantNEETCohortAfterMerge = pd.merge(PersistantNeetMoreThan12BFMerge,PersistantCensusRecordNEETwithLSOA,on='person_id',how='left')
#NEETStatusCohortAfterMerge.query("CurrentActivityCode != 619")
PersistantNEETCohortAfterMerge


# In[ ]:


PersistantNEETCohortAfterMerge.info()


# In[ ]:


PersistantNEETCohortAfterMerge.to_csv(retINTERIM_DATA_LOCATION() + "PersistantNEETLSOA.csv", index=False)
df = PersistantNEETCohortAfterMerge.astype(str)
df.info()
df.to_gbq('yhcr-prd-phm-bia-core.CB_MYSPACE_RT.PersistantNEETLSOA', 
                 chunksize=None, # I have tried with several chunk sizes, it runs faster when it's one big chunk 
                 if_exists='replace'
                 )


# In[ ]:


ct1 = PersistantNEETCohortAfterMerge.person_id.unique()
len(ct1)


# In[ ]:


# Geospatial analysis
import geopandas as gpd
#import cartopy as ccrs
import contextily as cx

# Visualisation
from matplotlib import font_manager
import matplotlib.patches as mpatches
import matplotlib.ticker as mtick
from shapely.geometry import Point
import seaborn as sns

import kaleido
import plotly.express as px
import plotly.io as pio


# In[ ]:


# Plotting config
pio.orca.config.executable = '/path/to/orca'
pio.orca.config.save()

# Define palettes
pal_full = ["#71297d","#ff9505","#4281a4","#057652","#0e0004"]
pal_gender = dict(F="#71297D" , M="#FF9505")
pal_school = dict(Primary="#71297D" , Secondary="#FF9505")
pal_auth_unauth = dict(Authorised="#71297D" , Unauthorised="#FF9505")

# Print palette
sns.palplot(sns.color_palette(pal_full))

plt.rcParams['axes.prop_cycle'] = plt.cycler(color=pal_full)

# Set Seaborn defaults
sns.set(
        rc={
 'axes.axisbelow': False,
 'axes.edgecolor': 'lightgrey',
 'axes.facecolor': 'None',
 'axes.grid': True,
 'grid.color': 'lightgrey',
 'axes.axisbelow': True,
 'axes.labelcolor': 'black',
 'axes.spines.right': False,
 'axes.spines.top': False,
 'figure.facecolor': 'white',
 'lines.solid_capstyle': 'round',
 'patch.edgecolor': 'w',
 'patch.force_edgecolor': True,
 'text.color': 'black',
 'xtick.bottom': False,
 'xtick.color': 'black',
 'xtick.direction': 'out',
 'xtick.top': False,
 'ytick.color': 'black',
 'ytick.direction': 'out',
 'ytick.left': False,
 'ytick.right': False,
}
)

sns.set_context("notebook", rc={"font.size":18,
                                "axes.titlesize":22,
                                "axes.labelsize":20})


# In[ ]:


# Matplotlib/Seaborn custom font check
font_manager.findSystemFonts(fontpaths=None, fontext="ttf")
font_manager.findfont("Plus Jakarta Sans") # Test with "Special Elite" too


# <div class="alert alert-block alert-success">
# <h2> # Filtering records pertaining to Bradford only ans Age of the student when they were 16 with LSOA at 15 #</h2>
# </div>
# 
# 

# In[ ]:


# PersistantDFE16Brasford = PersistantNEETCohortAfterMerge.query('LSOA_name.str.contains("Bradford") & Age == 16', engine='python')
# PersistantDFE16Brasford

PersistantDFE16Brasford = PersistantNEETCohortAfterMerge.query('LSOA_name.str.contains("Bradford") & Age >= 16', engine='python')
PersistantDFE16Brasford


# In[ ]:


# PersistantDFE16Brasford = PersistantDFE16Brasford.groupby(['academic_start_date','LSOA'], as_index=False).agg({
#         # 'academic_start_date':'max',
#         # 'LSOA':'count',
#         'LSOA_name':'max',
#         'geometry_home':'max',
#         'total_neet_observations': 'sum',
#         'custody_offender_observations': 'sum',
#         'status_unknown_observations': 'sum',
#         'CountofPeople': 'nunique'}).assign(NEETObservationRatio = lambda df: df["total_neet_observations"] / df["CountofPeople"],
#                                          CustodyOffenderRatio = lambda df: df["custody_offender_observations"] / df["CountofPeople"],
#                                          StatusUnknownRatio = lambda df: df["status_unknown_observations"] / df["CountofPeople"] )


# In[ ]:


PersistantDFE16Brasfordgp = PersistantDFE16Brasford.groupby(['person_id','academic_start_date'], as_index=False).agg({
        'LSOA':'max',
        'LSOA_name':'max',
        'geometry_home':'max',
        'total_neet_observations': 'sum',
        'custody_offender_observations': 'sum',
        'status_unknown_observations': 'sum'})

PersistantDFE16Brasfordgp

#PersistantDFE16Brasfordgp = PersistantDFE16Brasfordgp..drop_duplicates()
#PersistantDFE16Brasford.drop(['CurrentActivityCode'])
#PersistantDFE16Brasford.groupby(['LSOA'], as_index=False).agg({
        # 'academic_start_date':'max',
        # 'LSOA':'count',


# In[ ]:


PersistantDFE16Brasfordgp["CountofPeople"] = PersistantDFE16Brasfordgp["person_id"] 



# In[ ]:


PersistantDFE16Brasfordgp = PersistantDFE16Brasfordgp.groupby(['LSOA'], as_index=False).agg({
#         # 'academic_start_date':'max',
#         # 'LSOA':'count',
        'LSOA_name':'max',
        'geometry_home':'max',
        'total_neet_observations': 'sum',
        'custody_offender_observations': 'sum',
        'status_unknown_observations': 'sum',
        'CountofPeople': 'count'}).assign(NEETObservationRatio = lambda df: df["total_neet_observations"] / df["CountofPeople"],
                                         CustodyOffenderRatio = lambda df: df["custody_offender_observations"] / df["CountofPeople"],
                                         StatusUnknownRatio = lambda df: df["status_unknown_observations"] / df["CountofPeople"] )


# In[ ]:


#DFE16Brasford['LSOA'].value_counts().index[:10]
PersistantDFE16Brasfordgp


# In[ ]:


#for key in DataFrameDict.keys():
    
fig, ax = plt.subplots(figsize=(16,9))

plt.rcParams['font.size'] = 6
    # Create swarm plot
sns.swarmplot(
    #data=DataFrameDict[key], 
    data=PersistantDFE16Brasfordgp,
    x='LSOA', 
    y='CountofPeople', 
    hue=None, 
    alpha=.75,
    color=pal_full[0],
    size=6.6,
    dodge=True
)

    # Chart formatting

for label in ax.xaxis.get_ticklabels():
    label.set_rotation(45)

ax.set_title("LSOA Count of people being NEET", fontsize=24)
ax.set_ylabel("No of NEET Observations", fontsize=16)
ax.set_xlabel('', fontsize=22)
ax.set_xticklabels(ax.get_xticklabels(), fontsize=10)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=10)
#ax.yaxis.set_major_formatter('{x:.0}')

sns.despine(left=True)
#plt.savefig(nccis.returnPlotLocation() + "NEET16to18/LSOA_Neet_Count_swarmplot_" + str(key), dpi=300)

plt.savefig(returnPlotLocation() + "PersistantNEET16to18/LSOA_Neet_Count_swarmplot", dpi=300)


# In[ ]:


# # Geometry cretion
# # Assign the geometry column as the active geometry

PersistantDFE16Brasfordgp['geometry'] = gpd.GeoSeries.from_wkt(PersistantDFE16Brasfordgp['geometry_home'], crs=4258)
gdf_NEET_by_lsoa = gpd.GeoDataFrame(PersistantDFE16Brasfordgp, geometry='geometry')
gdf_NEET_by_lsoa = gdf_NEET_by_lsoa.set_index('LSOA_name')

gdf_NEET_by_lsoa = gdf_NEET_by_lsoa.sort_values("LSOA")
gdf_NEET_by_lsoa


# In[ ]:


# Draw the map for each LSOA in the dataframe
fig = px.choropleth_mapbox(gdf_NEET_by_lsoa,
                        geojson=gdf_NEET_by_lsoa.geometry,
                        locations=gdf_NEET_by_lsoa.index,
                        hover_data=['CountofPeople','LSOA'],
                        color="CountofPeople",
                        mapbox_style="carto-positron",
                        opacity=0.6,
                        color_continuous_scale="blackbody_r",
                        zoom=10.35,
                        center = {"lat": 53.7999, "lon": -1.7564},
                        width=1000,
                        height=900,
                        animation_frame = "LSOA",

)

fig.update_geos(fitbounds="locations", visible=False)

fig.update_traces(
    marker_line_width=0.7,
    marker_line_color="grey",
)


fig.update_layout(
    coloraxis_colorbar_title_text = 'Count of People',
    coloraxis_colorbar_tickformat="",
    title="Persistant NEET Count vs LSOA"
)

fig.write_html(returnPlotLocation() + "PersistantNEET16to18/maps/PersistantNEET16to18_lsoa.html")
fig.write_image(returnPlotLocation() + "PersistantNEET16to18/maps/PersistantNEET16to18_lsoa_plotly_map.svg")

fig.show()


# <!-- ## Simpson's Diversity Index is commonly used to measure the diversity of species in a biological community. However, it can be adapted to other contexts, including geographical areas like Lower Layer Super Output Areas (LSOAs). The formula for Simpson's Diversity Index is as follows:
# 
# ![Screenshot 2023-12-11 at 17.41.08.png](attachment:b69069e0-9fab-40b3-9d46-3499fd24c3fe.png)
# 
# 
# where:
# D is the Simpson's Diversity Index,
# s is the number of different types of LSOAs (species, in the ecological context),
# n is the number of LSOAs of the i-th type,
# N is the total number of LSOAs.
# Here's how you can calculate Simpson's Diversity Index for geographical LSOA data:
# 
# ![Screenshot 2023-12-11 at 17.44.37.png](attachment:ac0dc89e-728a-4630-826c-bbc46563f62a.png)
#  -->

# In[ ]:


censusRecordsPersistantNEET["LSOACount"] = censusRecordsPersistantNEET["LSOA"]
#censusRecordsPersistantNEET["CountofPeople"]=censusRecordsPersistantNEET["person_id"]


# In[ ]:


simpsonDiversityIndexCalc = censusRecordsPersistantNEET.groupby(["LSOA"], as_index=False).agg({
        'LSOACount': 'count'})
simpsonDiversityIndexCalc


# In[ ]:


# simpsonDiversityIndexCalc = censusRecordsPersistantNEET.groupby(["LSOA"], as_index=False).agg({
#         'LSOACount': 'count'})
# sDict = simpsonDiversityIndexCalc.to_dict(orient="records")
# sDict
#SDict = simpsonDiversityIndexCalc.to_dict('split')
#print(SDict['data'])

studentDict = simpsonDiversityIndexCalc.set_index('LSOA').to_dict()['LSOACount']



# <div class="alert alert-block alert-success">
# <b> # Use these functions below to access the Dataframe of interest </b><br>
# 1. everNEETObservation<br>
# 2. PersistantNeetMoreThan12<br>
# 3. notInCensusPeople<br>
# 4. peopleNotInCensusLSOA15<br>
# </div>
