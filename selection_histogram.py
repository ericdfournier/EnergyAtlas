#%% package imports

import numpy as np
import pandas as pd

from bokeh.layouts import (
    row, 
    column,
    widgetbox
    )
from bokeh.models import (
    ColumnDataSource,
    BoxSelectTool, 
    LassoSelectTool,
    HoverTool,
    LogColorMapper,
    Spacer,
    Select,
    DataRange1d,
    PreText
    )
from bokeh.plotting import (
    figure, 
    curdoc,
    )
from bokeh.palettes import Viridis6 as palette

#%% load map datasource

path = "/Users/edf/Repositories/EnergyAtlas/data/la_county_neighborhood_boundaries_single.json"
map_source = pd.read_json(path, typ='series', orient='column')
neighborhoods = {}

#%% clean raw map datasource

for i,feature in enumerate(map_source.features):
    
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

#%% Prep Source Data for Plotting

neighborhood_lon = [neighborhood["lon"] for neighborhood in neighborhoods.values()]
neighborhood_lat = [neighborhood["lat"] for neighborhood in neighborhoods.values()]
neighborhood_names = [neighborhood['name'] for neighborhood in neighborhoods.values()]
color_mapper = LogColorMapper(palette=palette)

#%% Generate Random Energy Data for Visualization

seed = (12345)
np.random.seed(seed)

size = 10000
mu = np.log([2000, 200])
sigma =[[0.2,0],[0,0.2]]

x_rnd,y_rnd = np.exp(np.random.multivariate_normal(mu, sigma, size).T)
x_rnd_y_rnd = np.divide(x_rnd,y_rnd)
year_rnd = np.random.randint(1850,2010,size)

unique_names = list(set(neighborhood_names))
neighborhoods = [None]*size

# TODO: FIX DEPRECATED INDEXING BELOW
for i in range(size):
    ind = np.random.randint(0,len(unique_names),1,dtype=int)
    neighborhoods[i] = unique_names[ind]

neigh_avg = np.zeros(len(unique_names),dtype=float)
for j in range(len(neigh_avg)):
    ind = [i for i, x in enumerate(neighborhoods) if x == unique_names[j]]
    neigh_avg[j] = np.average(y_rnd[ind])
    
#%% create reference data frame

df = pd.DataFrame()

df['size'] = x_rnd
df['consumption'] = y_rnd
df['intensity'] = x_rnd_y_rnd
df['year'] = year_rnd

#%% create axis map

axis_map = {
    "Energy Consumption [MBTU]": "consumption",
    "Building Size [sq.ft.]": "size",
    "Energy Consumption Intensity [MBTU / sq.ft.]" : "intensity",
    "Construction Vintage [Year]" : "year"
}

#%% create input controls

x_axis = Select(title='X-Axis', value='Building Size [sq.ft.]', options=sorted(axis_map.keys()))
y_axis = Select(title='Y-Axis', value='Energy Consumption [MBTU]', options=sorted(axis_map.keys()))

#%% create map columnar data source

map_source = ColumnDataSource(data=dict(
    lon = neighborhood_lon,
    lat = neighborhood_lat,
    avg_consumption = neigh_avg,
    name = neighborhood_names,
))

#%% create plot columnar data source

plot_source = ColumnDataSource(data=dict(
    x = df[axis_map[x_axis.value]].values,
    y = df[axis_map[y_axis.value]].values
))

#%% create stats

stats = PreText(text='', width=500)

#%% create map plot

MAP_TOOLS="pan,wheel_zoom,reset,hover,save"

m = figure(title="Los Angeles County Neighborhoods", tools=MAP_TOOLS, plot_width=450, plot_height=550, min_border=10, x_axis_location=None, y_axis_location=None,webgl=True)
m.grid.grid_line_color = None

mp = m.patches('lon','lat', source=map_source,
          fill_color={'field':'avg_consumption','transform':color_mapper},
          line_color="white", line_width=0.5)
          
hover = m.select_one(HoverTool)
hover.point_policy = "follow_mouse"
hover.tooltips = [
    ("Name", "@name"),
    ("(Long, Lat)", "($x, $y)"),
    ("Mean Annual Energy Consumption [MBTU / Megaparcel]", "@avg_consumption")
]

#%% create the scatter plot

PLOT_TOOLS="pan,wheel_zoom,box_zoom,box_select,lasso_select,reset"

p = figure(tools=PLOT_TOOLS, plot_width=600, plot_height=600, min_border=50, 
       min_border_left=50, toolbar_location="right", 
       x_axis_location="above", y_axis_location="left", 
       title="Linked Histograms", webgl=True)

p.background_fill_color = None
p.select(BoxSelectTool).select_every_mousemove = False
p.select(LassoSelectTool).select_every_mousemove = False    

r = p.scatter('x', 'y', source=plot_source, size=3, color="#3A5785", alpha=0.6, selection_color="orange")

# %% create the horizontal histogram

x = np.array(plot_source.data['x'])

hhist, hedges = np.histogram(x, bins=20)
hzeros = np.zeros(len(hedges)-1)
hmax = max(hhist)*1.1

LINE_ARGS_1 = dict(color="orange", line_color=None)
LINE_ARGS_2 = dict(color="#3A5785", line_color=None)

ph = figure(toolbar_location=None, plot_width=p.plot_width, 
            plot_height=200, x_range=p.x_range, y_range=(-hmax, hmax), 
            min_border=10, min_border_left=50, y_axis_location="right", 
            webgl=True)

ph.yaxis.major_label_orientation = np.pi/4
ph.xaxis.axis_label = x_axis.value

hh0 = ph.quad(bottom=0, left=hedges[:-1], right=hedges[1:], top=hhist, 
        color="white", line_color="#3A5785")
hh1 = ph.quad(bottom=0, left=hedges[:-1], right=hedges[1:], top=hzeros, 
              alpha=0.5, **LINE_ARGS_1)
hh2 = ph.quad(bottom=0, left=hedges[:-1], right=hedges[1:], top=hzeros, 
              alpha=0.1, **LINE_ARGS_2)

#%% create the vertical histogram

y = np.array(plot_source.data['y'])

vhist, vedges = np.histogram(y, bins=20)
vzeros = np.zeros(len(vedges)-1)
vmax = max(vhist)*1.1

pv = figure(toolbar_location=None, plot_width=200, 
            plot_height=p.plot_height, x_range=(-vmax, vmax), 
            y_range=p.y_range, min_border=10, y_axis_location="right", 
            webgl=True)

pv.xaxis.major_label_orientation = np.pi/4
pv.yaxis.axis_label = y_axis.value

vh0 = pv.quad(left=0, bottom=vedges[:-1], top=vedges[1:], right=vhist, 
        color="white", line_color="#3A5785")
vh1 = pv.quad(left=0, bottom=vedges[:-1], top=vedges[1:], right=vzeros, 
              alpha=0.5, **LINE_ARGS_1)
vh2 = pv.quad(left=0, bottom=vedges[:-1], top=vedges[1:], right=vzeros, 
              alpha=0.1, **LINE_ARGS_2)

#%% Update Major X Histograms

def update_major_x_histogram(source):
    
    hhist0, hedges0 = np.histogram(np.array(source.data['x']), bins=20)
    
    hh0.data_source.data['right'] = hedges0[1:]
    hh0.data_source.data['left'] = hedges0[:-1]
    hh0.data_source.data['top'] = hhist0

    hmax0 = max(hhist0)
    hmin0 = min(hhist0)    
    space = 0.1
    
    ph.set(y_range = DataRange1d(hmax0, space, hmin0))
    
#%% Update Major Y Histograms

def update_major_y_histogram(source):
    
    vhist0, vedges0 = np.histogram(np.array(source.data['y']), bins=20)

    vh0.data_source.data['top'] = vedges0[1:]
    vh0.data_source.data['bottom'] = vedges0[:-1]
    vh0.data_source.data['right'] = vhist0

    vmax0 = max(vhist0)
    vmin0 = min(vhist0)
    space = 0.1
    
    pv.set(x_range = DataRange1d(vmax0, space, vmin0))

#%% Update X-axis
    
def update_x_axis(attr, old, new):
                
    plot_source.data['x'] = df[axis_map[new]].values
    
    minx = min(plot_source.data['x'])
    maxx = max(plot_source.data['x'])
    space = 0.1
    
    p.set(x_range = DataRange1d(maxx, space, minx))
    ph.set(x_range = p.x_range)
    ph.xaxis.axis_label = new
    
    update_major_x_histogram(plot_source)
    update_minor_x_histogram(inds)
    
#%% Update Y-axis

def update_y_axis(attr, old, new):
    
    plot_source.data['y'] = df[axis_map[new]].values
    
    miny = min(plot_source.data['y'])
    maxy = max(plot_source.data['y'])
    space = 0.1
    
    p.set(y_range = DataRange1d(maxy, space, miny))
    pv.set(y_range = p.y_range)
    pv.yaxis.axis_label = y_axis.value

    update_major_y_histogram(plot_source)
    update_minor_y_histogram(inds)

#%% Update Stats Panel

def update_stats(inds):
        
    if len(inds) == 0 or len(inds) == len(df['size'].values):
        stats.text = str(df.describe())
    else:
        stats.text = str(df.ix[inds].describe())

#%% Upated Minor x Histogram

def update_minor_x_histogram(inds):
        
    if len(inds) == 0 or len(inds) == len(df['size'].values):
        hhist1, hhist2 = hzeros, hzeros
    else:
        neg_inds = np.ones_like(np.array(plot_source.data['x']), dtype=np.bool)
        neg_inds[inds] = False
        x = np.array(plot_source.data['x'])
        hhist1, _ = np.histogram(x[inds], bins=hedges)
        hhist2, _ = np.histogram(x[neg_inds], bins=hedges)

    hh1.data_source.data["top"] =  hhist1
    hh2.data_source.data["top"] = -hhist2
        
#%% Update Minor y Histogram

def update_minor_y_histogram(inds):
        
    if len(inds) == 0 or len(inds) == len(df['size'].values):
        vhist1, vhist2 = vzeros, vzeros
    else:
        neg_inds = np.ones_like(np.array(plot_source.data['y']), dtype=np.bool)
        neg_inds[inds] = False
        y = np.array(plot_source.data['y'])
        vhist1, _ = np.histogram(y[inds], bins=vedges)
        vhist2, _ = np.histogram(y[neg_inds], bins=vedges)

    vh1.data_source.data["right"] =  vhist1    
    vh2.data_source.data["right"] = -vhist2

#%% Update Map

#def update_map(inds):
#    
#    if len(inds) < 100:
#        # Decolor map
#    else:
#       # Color map on the basis of selected points
#       print('test')
        
#%% Update Selection    
    
def update_selection(attr, old, new):
            
    inds = np.array(new['1d']['indices'], dtype=int)
        
    update_stats(inds)
    update_minor_x_histogram(inds)
    update_minor_y_histogram(inds)
    
    #%% Create Selection Widgets

sizing_mode = 'fixed' 

widgets = widgetbox([x_axis, y_axis], sizing_mode=sizing_mode)
layout = row(column(widgets,stats,m), column(row(p, pv), row(ph, Spacer(width=200, height=200))))

curdoc().add_root(layout)
curdoc().title = "Selection Histogram"

x_axis.on_change('value', update_x_axis)
y_axis.on_change('value', update_y_axis)
r.data_source.on_change('selected', update_selection)

inds = np.array([],dtype=int)

# TODO: Integrate slider functionality into selection update callback

#min_vintage.on_change('value', update_selection)
#max_vintage.on_change('value', update_selection)

