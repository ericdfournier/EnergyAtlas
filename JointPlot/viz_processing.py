#%% Package Imports

import numpy as np
import pandas as pd
import itertools as it
import scipy.optimize as opt
from bokeh.models import (
    ColumnDataSource)

#%% Import Raw Data

path = "/Users/edf/Repositories/EnergyAtlas/JointPlot/raw/Malibu_Effect_Visualization_Megaparcels.txt"
df_new = pd.read_csv(path)

#%% Generate Total Consumption in MBTU

mbtu_per_kwh = 3412.14/1000000
mbtu_per_therm = 99976.1/1000000

df_new['CONSUMPTION'] = np.floor(np.add(np.multiply(df_new['sum_kwh'],mbtu_per_kwh),np.multiply(df_new['sum_therms'],mbtu_per_therm)))
df_new['INTENSITY'] = np.divide(df_new['CONSUMPTION'],df_new['SQFT'])

#%% Remove Outliers

def reject_outliers(sr, iq_range=0.5):
    pcnt = (1 - iq_range) / 2
    qlow, median, qhigh = sr.dropna().quantile([pcnt, 0.50, 1-pcnt])
    iqr = qhigh - qlow
    ind = (sr - median).abs() <= iqr
    out = np.zeros(len(sr),dtype=bool)
    out[ind] = True
    return ind

#%% Remove Spurrious Data Points

cons_check = df_new['CONSUMPTION'] > 0 & ~np.isnan(df_new['CONSUMPTION']) & ~np.isinf(df_new['CONSUMPTION'])
ints_check = df_new['INTENSITY'] > 0 & ~np.isnan(df_new['INTENSITY']) & ~np.isinf(df_new['INTENSITY'])
sqft_check = df_new['SQFT'] > 0 & ~np.isnan(df_new['SQFT']) & ~np.isinf(df_new['SQFT'])
year_check = df_new['YEAR'] > 0 & ~np.isnan(df_new['YEAR']) & ~np.isinf(df_new['YEAR'])
out1_check = reject_outliers(df_new['CONSUMPTION'],iq_range=0.99)
out2_check = reject_outliers(df_new['INTENSITY'],iq_range=0.99)

all_check = cons_check & ints_check & sqft_check & year_check & out1_check & out2_check

df_new = df_new.ix[all_check]

#%% Round Square Footage Values to the Nearest 100 sq.ft.

df_new['SQFT'] = np.ceil(df_new['SQFT']/100)*100

#%% Group by Neighborhood

#TODO: DUHH...The grouping is currently by CITY_NAME, this needs to be changed
# so that it is by NEIGHBORHOOD_NAME. There are only 86 cities and 296 
# neighborhoods...

neighborhood_mean_usage = df_new.groupby('CITY_NAME').CONSUMPTION.agg('mean')
neighborhood_mean_intensity = df_new.groupby('CITY_NAME').INTENSITY.agg('mean')
neighborhood_mean_sqft = df_new.groupby('CITY_NAME').SQFT.agg('mean')
neighborhood_count = df_new.groupby('CITY_NAME').CONSUMPTION.agg('count')

#%% Generate Output Statistics Data frames

neighborhood_stats = pd.DataFrame(index=neighborhood_mean_usage.index)
neighborhood_stats['mean_usage'] = neighborhood_mean_usage
neighborhood_stats['mean_intensity'] = neighborhood_mean_intensity
neighborhood_stats['mean_sqft'] = neighborhood_mean_sqft
neighborhood_stats['count'] = neighborhood_count

#%% Generate Full Output Data Frames

final = pd.DataFrame(index=range(len(df_new)))

final['size'] = np.asarray(df_new['SQFT']).astype(int)
final['consumption'] = np.asarray(df_new['CONSUMPTION']).astype(int)
final['intensity'] = np.asarray(df_new['INTENSITY']).astype(float)
final['year'] = np.asarray(df_new['YEAR']).astype(int)
final['name'] = np.asarray(df_new['CITY_NAME'])

#%% Generate Medium Output Data Frame

size = 100000
choices = range(len(final))
inds = np.random.choice(choices,size,replace=False)
final_medium = final.ix[inds,:]
final_medium.reset_index(inplace=True)
final_medium.drop('index',axis=1,inplace=True)

#%% Generate Polynomial Fits

inds = it.product([0,1,2,3],repeat=2)
fits = dict()

def func(x,a,b,c):
    return a+(b/(x+c))

for i,v in enumerate(inds):
    x = final_medium.ix[:,v[0]]
    y = final_medium.ix[:,v[1]]
    if v == (0,2) or v == (2,0):
        fits[v], _ = opt.curve_fit(func, x, y)
    else:
        fits[v] = np.poly1d(np.polyfit(x, y, 1))
        
#%% Generate Test Plots

inds = it.product([0,1,2,3],repeat=2)
yhat = pd.DataFrame(np.zeros([100,16]))
xp = pd.DataFrame(np.zeros([100,16]))
names = []

for i,v in enumerate(inds):
    names.append(v)
    xp.ix[:,i] = np.linspace(min(final_medium.ix[:,v[0]]), max(final_medium.ix[:,v[0]]), 100)
    if v == (0,2) or v == (2,0):
        yhat.ix[:,i] = func(xp.ix[:,i], fits[v][0], fits[v][1], fits[v][2])
    else:
        yhat.ix[:,i] = fits[v](xp.ix[:,i])
        
yhat.columns = names
xp.columns = names

#%% Load Static Map Datasource

map_path = "/Users/edf/Repositories/EnergyAtlas/JointPlot/data/json/la_county_neighborhood_boundaries_single_simple.json"
map_source = pd.read_json(map_path, typ='series', orient='column')
neighborhoods = {}

#%% Clean Raw Static Map Datasource

for i,feature in enumerate(map_source.features):
        
    if feature['geometry']['type'] == 'Polygon':
        
        coords = feature['geometry']['coordinates'][0]
        name = feature['properties']['Name'][:]
        lon,lat,_ = np.reshape(coords,[len(coords),3]).T
        neighborhoods[i] = {}
        neighborhoods[i]['lat'] = lat
        neighborhoods[i]['lon'] = lon
        neighborhoods[i]['name'] = name
        
    elif feature['geometry']['type'] == 'MultiPolygon':
        
        coords = feature['geometry']['coordinates'][:]
        name = feature['properties']['Name'][:]
        lons = []
        lats = []
        
        for j, poly in enumerate(coords):
            
            lon,lat,_ = np.reshape(coords[j][0],[len(coords[j][0]),3]).T
            lons.append(lon)
            lats.append(lat)
        
        neighborhoods[i] = {}
        neighborhoods[i]['lat'] = lats
        neighborhoods[i]['lon'] = lons
        neighborhoods[i]['name'] = name

#%% Prep Static Map Source Data for Plotting

neighborhood_lon = np.asarray([neighborhood["lon"] for neighborhood in neighborhoods.values()])
neighborhood_lat = np.asarray([neighborhood["lat"] for neighborhood in neighborhoods.values()])
neighborhood_names = np.asarray([neighborhood['name'] for neighborhood in neighborhoods.values()])
neighborhood_avg_usage = np.zeros(len(neighborhood_names)) 
neighborhood_avg_intensity = np.zeros(len(neighborhood_names)) 
neighborhood_avg_sqft = np.zeros(len(neighborhood_names))   
neighborhood_count = np.zeros(len(neighborhood_names))   

for i,n in enumerate(neighborhood_names):
    if any(np.in1d(neighborhood_stats.index.values, n)) == 0:
        neighborhood_avg_usage[i] = np.nan
        neighborhood_avg_intensity[i] = np.nan
        neighborhood_avg_sqft[i] = np.nan
        neighborhood_count = np.nan
    else:
        neighborhood_avg_usage[i] = neighborhood_stats.loc[str(n)].mean_usage
        neighborhood_avg_intensity[i] = neighborhood_stats.loc[str(n)].mean_intensity
        neighborhood_avg_sqft[i] = neighborhood_stats.loc[str(n)].mean_sqft
        neighborhood_count = neighborhood_stats.loc[str(n)].count
   
color = np.asarray(["white"]*len(neighborhood_names))
alpha = np.asarray([1.0]*len(neighborhood_names))

#%% Create Map columnar data source

map_source = ColumnDataSource(data=dict(
    lon = neighborhood_lon,
    lat = neighborhood_lat,
    avg_consumption = neighborhood_avg,
    name = neighborhood_names,
    color = color,
    alpha = alpha
))             

#%% Export Data Frame to CSV

final.to_csv("/Users/edf/Repositories/EnergyAtlas/JointPlot/raw/input_table.csv")
final_medium.to_csv("/Users/edf/Repositories/EnergyAtlas/JointPlot/raw/input_table_medium.csv")
neighborhood_stats.to_csv("/Users/edf/Repositories/EnergyAtlas/JointPlot/raw/neighborhood_stats.csv")
yhat.to_csv("/Users/edf/Repositories/EnergyAtlas/JointPlot/raw/yhat.csv")
xp.to_csv("/Users/edf/Repositories/EnergyAtlas/JointPlot/raw/xp.csv")

#%% Export Dataframe to Pickle

final.to_pickle("/Users/edf/Repositories/EnergyAtlas/JointPlot/data/pkl/input_table.pkl")
final_medium.to_pickle("/Users/edf/Repositories/EnergyAtlas/JointPlot/data/pkl/input_table_medium.pkl")
neighborhood_stats.to_pickle("/Users/edf/Repositories/EnergyAtlas/JointPlot/data/pkl/neighborhood_stats.pkl")
yhat.to_pickle("/Users/edf/Repositories/EnergyAtlas/JointPlot/data/pkl/yhat.pkl")
xp.to_pickle("/Users/edf/Repositories/EnergyAtlas/JointPlot/data/pkl/xp.pkl")