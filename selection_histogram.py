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
    Range1d,
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

#%% create map columnar data source

map_source = ColumnDataSource(data=dict(
    lon = neighborhood_lon,
    lat = neighborhood_lat,
    avg_consumption = neigh_avg,
    name = neighborhood_names,
))

#%% create plot columnar data source

plot_source = ColumnDataSource(data=dict(
    x = [],
    y = []
))

#%% create stats

stats = PreText(text='', width=500)

#%% create map plot

def create_map(map_source):

    MAP_TOOLS="pan,wheel_zoom,reset,hover,save"
    
    m = figure(title="Los Angeles County Neighborhoods", tools=MAP_TOOLS, 
               plot_width=700, plot_height=900, min_border=10, 
               x_axis_location=None, y_axis_location=None,webgl=True)
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

    return m, mp

#%% create the scatter plot

def create_scatter(plot_source):

    PLOT_TOOLS="pan,wheel_zoom,box_zoom,box_select,lasso_select,reset"
    
    p = figure(tools=PLOT_TOOLS, plot_width=700, plot_height=700, min_border=50, 
           min_border_left=50, toolbar_location="right", 
           x_axis_location="above", y_axis_location="left", 
           title="Linked Histograms", webgl=True)
    
    p.background_fill_color = None
    p.select(BoxSelectTool).select_every_mousemove = False
    p.select(LassoSelectTool).select_every_mousemove = False  
    
    p.circle('x', 'y', source=plot_source, size=3, color="#3A5785", alpha=0.6, 
             selection_color="orange", nonselection_alpha=0.1, selection_alpha=0.4)

    return p

# %% create the horizontal histogram

def create_major_x_histogram(plot_source):

    LINE_ARGS_1 = dict(color="orange", line_color=None)
    LINE_ARGS_2 = dict(color="#3A5785", line_color=None)
    
    hhist0, hedges0 = np.histogram(np.array(plot_source.data['x']), bins=20)
    hzeros0 = np.zeros(len(hedges0)-1)
    hmax0 = max(hhist0)*1.1
    
    ph = figure(toolbar_location=None, plot_width=p.plot_width, 
                plot_height=200, x_range=p.x_range, y_range = Range1d(end=hmax0,start=-hmax0),
                min_border=10, min_border_left=50, y_axis_location="right", 
                webgl=True)
    
    ph.yaxis.major_label_orientation = np.pi/4
    ph.xaxis.axis_label = x_axis.value
    
    hh0 = ph.quad(bottom=0, left=hedges0[:-1], right=hedges0[1:], top=hhist0, 
            color="white", line_color="#3A5785")
    hh1 = ph.quad(bottom=0, left=hedges0[:-1], right=hedges0[1:], top=hzeros0, 
                  alpha=0.5, **LINE_ARGS_1)
    hh2 = ph.quad(bottom=0, left=hedges0[:-1], right=hedges0[1:], top=hzeros0, 
                  alpha=0.1, **LINE_ARGS_2)

    return ph, hh0, hh1, hh2, hhist0, hedges0

#%% create the vertical histogram

def create_major_y_histogram(plot_source):
    
    LINE_ARGS_1 = dict(color="orange", line_color=None)
    LINE_ARGS_2 = dict(color="#3A5785", line_color=None)

    vhist0, vedges0 = np.histogram(np.array(plot_source.data['y']), bins=20)
    vzeros0 = np.zeros(len(vedges0)-1)
    vmax0 = max(vhist0)*1.1
    
    pv = figure(toolbar_location=None, plot_width=200, 
                plot_height=p.plot_height, x_range = Range1d(end=vmax0, start=-vmax0),
                y_range=p.y_range, min_border=10, y_axis_location="right", 
                webgl=True)
    
    pv.xaxis.major_label_orientation = np.pi/4
    pv.yaxis.axis_label = y_axis.value
    
    vh0 = pv.quad(left=0, bottom=vedges0[:-1], top=vedges0[1:], right=vhist0, 
            color="white", line_color="#3A5785")
    vh1 = pv.quad(left=0, bottom=vedges0[:-1], top=vedges0[1:], right=vzeros0, 
                  alpha=0.5, **LINE_ARGS_1)
    vh2 = pv.quad(left=0, bottom=vedges0[:-1], top=vedges0[1:], right=vzeros0, 
                  alpha=0.1, **LINE_ARGS_2)

    return pv, vh0, vh1, vh2, vhist0, vedges0
    
#%% Update Major X Histograms

def update_major_x_histogram(plot_source):
    
    hhist0, hedges0 = np.histogram(np.array(plot_source.data['x']), bins=20)
    
    hmax0 = max(hhist0)*1.1
    ph.y_range = Range1d(end=hmax0,start=-hmax0)
    
    hh0.data_source.data['right'] = hedges0[1:]
    hh0.data_source.data['left'] = hedges0[:-1]
    hh0.data_source.data['top'] = hhist0
    
    update_minor_x_histogram(plot_source, hedges0, inds)
    
#%% Update Major Y Histograms

def update_major_y_histogram(plot_source):
    
    vhist0, vedges0 = np.histogram(np.array(plot_source.data['y']), bins=20)

    vmax0 = max(vhist0)*1.1
    pv.x_range = Range1d(end=vmax0,start=-vmax0)
    
    vh0.data_source.data['top'] = vedges0[1:]
    vh0.data_source.data['bottom'] = vedges0[:-1]
    vh0.data_source.data['right'] = vhist0

    update_minor_y_histogram(plot_source, vedges0, inds)

#%% Update X-axis
    
def update_x_axis(attr, old, new):
                
    plot_source.data['x'] = df[axis_map[new]].values

    maxx = max(plot_source.data['x'])*1.1
    minx = min(plot_source.data['x'])-(max(plot_source.data['x'])*0.1)

    p.x_range = Range1d(end=maxx,start=minx)

    ph.x_range = p.x_range
    ph.xaxis.axis_label = new
    
    update_major_x_histogram(plot_source)
    
#%% Update Y-axis

def update_y_axis(attr, old, new):
    
    plot_source.data['y'] = df[axis_map[new]].values

    maxy = max(plot_source.data['y'])*1.1
    miny = min(plot_source.data['y'])-(max(plot_source.data['y'])*0.1)
    
    p.y_range = Range1d(end=maxy,start=miny)
    
    pv.y_range = p.y_range
    pv.yaxis.axis_label = new

    update_major_y_histogram(plot_source)
        
#%% Update Stats Panel

def update_stats(inds):
        
    if len(inds) == 0 or len(inds) == len(df['size'].values):
        stats.text = str(df.describe())
    else:
        stats.text = str(df.ix[inds].describe())

#%% Upated Minor x Histogram

def update_minor_x_histogram(plot_source, hedges0, inds):
        
    if len(inds) == 0 or len(inds) == len(df['size'].values):
        hhist1, hhist2 = hzeros0, hzeros0
    else:
        neg_inds = np.ones_like(np.array(plot_source.data['x']), dtype=np.bool)
        neg_inds[inds] = False
        x = np.array(plot_source.data['x'])
        hhist1, _ = np.histogram(x[inds], bins=hedges0)
        hhist2, _ = np.histogram(x[neg_inds], bins=hedges0)

    hh1.data_source.data["top"] =  hhist1
    hh2.data_source.data["top"] = -hhist2
        
#%% Update Minor y Histogram

def update_minor_y_histogram(plot_source, vedges0, inds):
        
    if len(inds) == 0 or len(inds) == len(df['size'].values):
        vhist1, vhist2 = vzeros0, vzeros0
    else:
        neg_inds = np.ones_like(np.array(plot_source.data['y']), dtype=np.bool)
        neg_inds[inds] = False
        y = np.array(plot_source.data['y'])
        vhist1, _ = np.histogram(y[inds], bins=vedges0)
        vhist2, _ = np.histogram(y[neg_inds], bins=vedges0)

    vh1.data_source.data["right"] =  vhist1    
    vh2.data_source.data["right"] = -vhist2

#%% Update Selection    
    
def update_selection(attr, old, new):
            
    inds = np.array(new['1d']['indices'], dtype=int)
        
    update_stats(inds)
    update_minor_x_histogram(plot_source, hedges0, inds)
    update_minor_y_histogram(plot_source, vedges0, inds)
    
    return inds

#%% Update plot source data

def update_data():
    
    plot_source.data['x'] = df[axis_map[x_axis.value]].values
    plot_source.data['y'] = df[axis_map[y_axis.value]].values
        
#%% create axes selectors

x_axis = Select(title='X-Axis', value='Building Size [sq.ft.]', options=sorted(axis_map.keys()))
y_axis = Select(title='Y-Axis', value='Energy Consumption [MBTU]', options=sorted(axis_map.keys()))

#%% Generate Plot Variables

inds = np.asarray([], dtype=int)

update_data()

p = create_scatter(plot_source)
ph, hh0, hh1, hh2, hzeros0, hedges0 = create_major_x_histogram(plot_source)
pv, vh0, vh1, vh2, vzeros0, vedges0 = create_major_y_histogram(plot_source)
m, mp = create_map(map_source)

#%% Create Widgets and Layout

sizing_mode = 'fixed'
widgets = widgetbox([x_axis, y_axis], sizing_mode=sizing_mode)
layout = column(row(column(row(p, pv), row(ph, Spacer(width=200, height=200))),
                    column(Spacer(width=50)), column(m)), row(widgets,stats))
curdoc().add_root(layout)
curdoc().title = "Selection Histogram"

x_axis.on_change('value', update_x_axis)
y_axis.on_change('value', update_y_axis)
plot_source.on_change('selected', update_selection)

# TODO: Integrate slider functionality into selection update callback

#min_vintage.on_change('value', update_selection)
#max_vintage.on_change('value', update_selection)

