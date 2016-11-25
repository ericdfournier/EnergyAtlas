# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 10:40:14 2016

@author: efournier
"""

#%% Import Packages

import numpy as np
import pandas as pd

from bokeh.io import show
from bokeh.models import (
    ColumnDataSource,
    HoverTool,
    LogColorMapper
    )
from bokeh.plotting import (
    figure,
    output_file
    )
from bokeh.palettes import Viridis6 as palette

#%% Munge the Raw Data

path = "V:\\PIER_Data\\Eric_Fournier\\JSON\\la_county_neighborhood_boundaries_single.json"
raw_source = pd.read_json(path, typ='series', orient='column')

neighborhoods = {}

for i,feature in enumerate(raw_source.features):
    
    if feature['geometry']['type'] == 'Polygon':
        
        coords = feature['geometry']['coordinates'][0]
        name = feature['properties']['Name'][:]
        lon,lat,_ = np.reshape(coords,[len(coords),3]).T
        neighborhoods[i] = {}
        neighborhoods[i]['lat'] = lat
        neighborhoods[i]['lon'] = lon
        neighborhoods[i]['name'] = name
    
    # There is still an issue with the parsing of multi-part polygons
    
    elif feature['geometry']['type'] == 'MultiPolygon':
        
        coords = feature['geometry']['coordinates'][:]
        name = feature['properties']['Name'][:]
        lons = []
        lats = []
        
        for j, poly in enumerate(coords):
            
            lon,lat,_ = np.reshape(coords[j][0],[len(coords[j][0]),3]).T
            # There is a nesting list issue here that might be resolved in the future
            lons.append(lon)
            lats.append(lat)
        
        neighborhoods[i] = {}
        neighborhoods[i]['lat'] = lats
        neighborhoods[i]['lon'] = lons
        neighborhoods[i]['name'] = name
        
#%% Generate Random Energy Data for Visualization

energy_consumption = np.random.randint(0,10000,len(neighborhoods))     
        
#%% Prep Source Data for Plotting

neighborhood_xs = [neighborhood["lon"] for neighborhood in neighborhoods.values()]
neighborhood_ys = [neighborhood["lat"] for neighborhood in neighborhoods.values()]
neighborhood_names = [neighborhood['name'] for neighborhood in neighborhoods.values()]
color_mapper = LogColorMapper(palette=palette)

source = ColumnDataSource(data=dict(
    x = neighborhood_xs,
    y = neighborhood_ys,
    energy = energy_consumption,
    name = neighborhood_names,
))

#%% Generate Plot

TOOLS="pan,wheel_zoom,box_zoom,reset,hover,save"

p = figure(title="Los Angeles County Neighborhoods", tools=TOOLS, plot_width=900, plot_height=900, min_border=10, x_axis_location=None, y_axis_location=None,webgl=True)

p.grid.grid_line_color = None

p.patches('x', 'y', source=source,
          fill_color={'field':'energy','transform':color_mapper},
          line_color="white", line_width=0.5)
          
hover = p.select_one(HoverTool)
hover.point_policy = "follow_mouse"
hover.tooltips = [
    ("Name", "@name"),
    ("(Long, Lat)", "($x, $y)"),
    ("Mean Annual Energy Consumption [MBTU / Megaparcel]", "@energy")
]

output_file("V:\\PIER_Data\\Eric_Fournier\\HTML\\selection_map.html", title="selection_map.py example")

show(p)